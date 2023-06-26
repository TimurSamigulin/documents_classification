import os

from sklearn.model_selection import train_test_split
from transformers import LayoutLMv3FeatureExtractor, LayoutLMv3TokenizerFast, LayoutLMv3Processor, LayoutLMv3ForSequenceClassification
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

from app.model.document_dataset import DocumentClassificationDataset
from app.model.model_class import ModelModule
from torch.utils.data import DataLoader

def get_dataloader(train_images, test_images, doc_classes, processor, batch_size=8):
    train_dataset = DocumentClassificationDataset(train_images, doc_classes, processor)
    test_dataset = DocumentClassificationDataset(test_images, doc_classes, processor)
    
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    test_data_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    
    return train_data_loader, test_data_loader


def get_trainer(accelerator='gpu', devices=[0]):
    model_checkpoint = ModelCheckpoint(
        filename="{epoch}-{step}-{val_loss:.4f}", save_last=True, save_top_k=3, monitor="val_loss", mode="min"
    )
    epoch = 1
    logger = CSVLogger("logs")
    if accelerator == 'gpu':
        trainer = pl.Trainer(
            accelerator="gpu",
            precision=32,
            devices=devices,
            max_epochs=epoch,
            callbacks=[
                model_checkpoint
            ],
            logger=logger,
            # log_every_n_steps=1,
        )
    else:
        trainer = pl.Trainer(
            accelerator="cpu",
            precision=32,
            max_epochs=epoch,
            callbacks=[
                model_checkpoint
            ],
            logger=logger,
            # log_every_n_steps=1,
        )
    
    return trainer

def learn_model(files_df, accelerator='gpu', devices=[0]):
    # print(files_df)
    doc_classes = list(files_df['category'].unique())
    train_images, test_images = train_test_split(files_df, test_size=0.2)
    
    feature_extractor = LayoutLMv3FeatureExtractor(apply_ocr=False)
    tokenizer = LayoutLMv3TokenizerFast.from_pretrained("microsoft/layoutlmv3-base")
    processor = LayoutLMv3Processor(feature_extractor, tokenizer)
    
    train_data_loader, test_data_loader = get_dataloader(train_images, test_images, doc_classes, processor)
    
    model_module = ModelModule(len(doc_classes))

    trainer = get_trainer(accelerator, devices)
    
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    trainer.fit(model_module, train_data_loader, test_data_loader)
