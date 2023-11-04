import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from torch.utils.tensorboard import SummaryWriter

from dataset import BilingualDataset, casual_mask
from model import build_transformer
from pathlib import Path
from config import get_weights_file_path, get_config, latest_weights_file_path
from tqdm import tqdm


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

    for epoch in tqdm(range(config["num_epochs"])):
        model.train()
        for batch in tqdm(train_dataloader):
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

            writer.add_scalar("train loss", loss.item(), global_step=100)
            writer.flush()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save(
            {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()},
            model_filename,
        )


if __name__ == "__main__":
    config = get_config()
    train_model(config)
