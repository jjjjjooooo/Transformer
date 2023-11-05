import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from torch.utils.tensorboard import SummaryWriter
import torchmetrics

from dataset import BilingualDataset, casual_mask
from model import build_transformer
from pathlib import Path
from config import get_weights_file_path, get_config, latest_weights_file_path
from tqdm import tqdm


def greedy_decode(model, src, src_mask, tokenizer_tgt, config, device):
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")

    encoder_output = model.encode(src, src_mask).to(device)

    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(src).to(device)

    while True:
        if decoder_input.size(1) == config["seq_len"]:
            break

        decoder_mask = casual_mask(decoder_input.size(1)).type_as(src_mask).to(device)

        decoder_output = model.decode(
            tgt=decoder_input, encoder_output=encoder_output, src_mask=src_mask, tgt_mask=decoder_mask
        )

        prob = model.project(decoder_output[:, -1])

        _, next_word_token = torch.max(prob, dim=1)

        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(src).fill_(next_word_token.item()).to(device)], dim=1
        )

        if next_word_token == eos_idx:
            break

    return decoder_input.squeeze(0)


def run_validation(model, validation_ds, tokenizer_tgt, config, device, writer, global_step):
    model.eval()

    source_texts = []
    expected = []
    predicted = []

    with torch.no_grad():
        for batch in validation_ds:
            encoder_input = batch["encoder_input"].to(device)  # (batch, seq_len)
            encoder_mask = batch["encoder_mask"].to(device)  # (batch, 1, 1, seq_len)

            assert encoder_input.size(0) == 1, "Batch size is 1 for validation."

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_tgt, config, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

    if writer:
        # Evaluate the character error rate
        # Compute the char error rate
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar("validation cer", cer, global_step)
        writer.flush()

        # Compute the word error rate
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar("validation wer", wer, global_step)
        writer.flush()

        # Compute the BLEU metric
        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar("validation BLEU", bleu, global_step)
        writer.flush()


def get_all_sentences(dataset, language):
    for item in dataset:
        yield item["translation"][language]


def get_build_tokenizer(config, dataset, language):
    tokenizer_path = Path(config["tokenizer_file"].format(language))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(dataset, language), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_ds(config):
    # It only has the train split, so we divide it overselves
    ds_raw = load_dataset(config["datasource"], f"{config['src_lang']}-{config['tgt_lang']}", split="train")

    # Build tokenizers
    tokenizer_src = get_build_tokenizer(config, ds_raw, config["src_lang"])
    tokenizer_tgt = get_build_tokenizer(config, ds_raw, config["tgt_lang"])

    # Keep 90% for training, 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(
        train_ds_raw, tokenizer_src, tokenizer_tgt, config["src_lang"], config["tgt_lang"], config["seq_len"]
    )
    val_ds = BilingualDataset(
        val_ds_raw, tokenizer_src, tokenizer_tgt, config["src_lang"], config["tgt_lang"], config["seq_len"]
    )

    # Find the maximum length of each sentence in the source and target sentence
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item["translation"][config["src_lang"]]).ids
        tgt_ids = tokenizer_tgt.encode(item["translation"][config["tgt_lang"]]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"Max length of source sentence: {max_len_src}")
    print(f"Max length of target sentence: {max_len_tgt}")

    train_dataloader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config, src_vocab_size, tgt_vocab_size):
    model = build_transformer(src_vocab_size, tgt_vocab_size, config["seq_len"], config["seq_len"], config["model_dim"])
    return model


def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    writer = SummaryWriter(config["experiment_name"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id("[PAD]"))
    global_step = 0

    for epoch in range(config["num_epochs"]):
        for batch in tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}"):
            model.train()

            encoder_input = batch["encoder_input"].to(device)  # (batch_size, seq_len)
            decoder_input = batch["decoder_input"].to(device)  # (batch_size, seq_len)
            encoder_mask = batch["encoder_mask"].to(device)  # (batch_size, 1, 1, seq_len)
            decoder_mask = batch["decoder_mask"].to(device)  # (batch_size, 1, seq_len, seq_len)
            label = batch["label"].to(device)  # (batch_size, seq_len)

            encoder_output = model.encode(src=encoder_input, src_mask=encoder_mask)  # (batch_size, seq_len, model_dim)
            decoder_output = model.decode(
                tgt=decoder_input, encoder_output=encoder_output, src_mask=encoder_mask, tgt_mask=decoder_mask
            )  # (batch_size, seq_len, model_dim)
            projection_output = model.project(decoder_output)  # (batch_size, seq_len, tgt_vocab_size)

            # (batch_size, seq_len, tgt_vocab_size) -> (batch_size * seq_len, tgt_vocab_size)
            loss = loss_fn(projection_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))

            writer.add_scalar("train loss", loss.item(), global_step)
            writer.flush()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1

        # Run validation at the end of every epoch
        run_validation(
            model,
            val_dataloader,
            tokenizer_tgt,
            config,
            device,
            writer,
        )

        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
            },
            model_filename,
        )


if __name__ == "__main__":
    config = get_config()
    train_model(config)
