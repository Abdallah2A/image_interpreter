from zenml.pipelines import pipeline
from steps.data_loader import data_loader_step
from steps.model_definition import model_definition_step
from steps.training import training_step
from steps.evaluation import evaluation_step


@pipeline(enable_cache=False)
def coco_caption_pipeline():
    train_loader, val_loader, tokenizer = data_loader_step()
    model = model_definition_step(tokenizer)
    _, history = training_step(train_loader, val_loader, model, tokenizer)
    evaluation_step(history)


if __name__ == '__main__':
    coco_caption_pipeline().run()
