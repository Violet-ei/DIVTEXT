import datetime
from utils import *
import torch
from logger import get_logger
from training import Trainer
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, BertForSequenceClassification
from args import *

parser = get_parser()
args = parser.parse_args()

logger = get_logger(
    log_file=f"{args.embedding_type}_{args.mapping_strategy}_{args.privatization_strategy}_eps_{args.eps}_top_{args.top_k}_save_{args.save_stop_words}_{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.txt"
)
logger.info(f"{args.dataset}, args: {args}")

if __name__ == "__main__":
    # 给 PyTorch 的随机数发生器（CPU 和 GPU）统一设置随机种子，以确保实验的可复现性
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # 加载数据、映射和字典
    train_data, dev_data, test_data = load_data(args.dataset)
    sim_word_dict, p_dict = get_customized_mapping(eps=args.eps, top_k=args.top_k)

    # sanitized 数据的生成
    if args.privatization_strategy == "s1":
        train_data = generate_new_sents_s1(df=train_data, sim_word_dict=sim_word_dict, p_dict=p_dict, save_stop_words=args.save_stop_words)
        dev_data = generate_new_sents_s1(df=dev_data, sim_word_dict=sim_word_dict, p_dict=p_dict, save_stop_words=args.save_stop_words)

    train_dataset = Bert_dataset(train_data)
    dev_dataset = Bert_dataset(dev_data)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)

    logger.info(f"train_data:{len(train_data)}, dev_data:{len(dev_data)}")

    # ------------------------------
    # 5. 模型
    # ------------------------------
    model = BertForSequenceClassification.from_pretrained(
        args.model_type,
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False
    )

    optimizer = AdamW(model.parameters(), lr=args.lr, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_loader) * args.epochs
    )

    trainer = Trainer(
        model=model,
        scheduler=scheduler,
        optimizer=optimizer,
        n_epochs=args.epochs,
        log_steps=args.log_steps,
        eval_steps=args.eval_steps,
        use_cuda=args.use_cuda,
        logger=logger
    )

    # 训练，训练时评估 dev 集
    trainer.train(train_loader, dev_loader)  # 

    acc = trainer.predict(dev_loader)
    logger.info(f"✅ Final Dev Accuracy: {acc:.4f}")
