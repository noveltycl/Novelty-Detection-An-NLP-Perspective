import torch
import argparse
import os
import csv
import logging
import json
import copy
import datetime
import sys
import numpy as np
import pandas as pd
from numpy import cov 
logger = logging.getLogger(__name__)
import torch.nn as nn
import transformers
import random
from torch.utils.data import TensorDataset
from transformers import AutoTokenizer, AutoModelWithLMHead
from transformers import AutoModel,BertConfig
from transformers import (BertModel, BertPreTrainedModel)
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, f1_score
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr


PRETRAINED_MODEL_MAP = {'bert': BertModel}

fold = str(5)

logging.basicConfig(filename=str('Log/'+str(datetime.datetime.now()).replace(' ', '_'))+'BertModel.log',
					filemode='a',
					format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
					datefmt = '%m/%d/%Y %H:%M:%S',
					level = logging.INFO)
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class FCLayer(nn.Module):
	def __init__(self, input_dim, output_dim, dropout_rate=0., use_activation=True):
		super(FCLayer, self).__init__()
		self.use_activation = use_activation
		self.dropout = nn.Dropout(dropout_rate)
		self.linear = nn.Linear(input_dim, output_dim)
		self.tanh = nn.Tanh()

	def forward(self, x):
		x = self.dropout(x)
		if self.use_activation:
			x = self.tanh(x)
		return self.linear(x)


class NoveltyClassifier(BertPreTrainedModel):
	def __init__(self, config, args):
		super(NoveltyClassifier, self).__init__(config)
		self.bert = PRETRAINED_MODEL_MAP[args.model_type](config=config)
		self.num_labels = config.num_labels
		self.label_classifier = FCLayer(config.hidden_size, config.num_labels, args.dropout_rate, use_activation=False)


	def forward(self, input_ids, attention_mask, token_type_ids, labels):
		outputs = self.bert(input_ids, attention_mask=attention_mask,token_type_ids=token_type_ids)
		pooled_output = outputs[1]
		logits = self.label_classifier(pooled_output)

		outputs = (logits,) + outputs[2:]
		if labels is not None:
			if self.num_labels == 1:
				loss_fct = nn.MSELoss()
				loss = loss_fct(logits.view(-1), labels.view(-1))
			else:
				loss_fct = nn.CrossEntropyLoss()
				loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
			outputs = (loss,) + outputs
		return outputs

class InputExample(object):
	def __init__(self, guid, text_a,text_b,label):
		self.guid = guid
		self.text_a = text_a
		self.text_b = text_b
		self.label = label

	def __repr__(self):
		return str(self.to_json_string())

	def to_dict(self):
		output = copy.deepcopy(self.__dict__)
		return output

	def to_json_string(self):
		return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

# def pearson_cf(y_true, y_pred):
# 		a = y_true - K.mean(y_true)
# 		b = y_pred - K.mean(y_pred)
# 		num = K.sum(a * b)
# 		den = K.sqrt(K.sum(a**2) * K.sum(b**2))
# 		return (num/(den+0.00001))

class InputFeatures(object):
	def __init__(self, input_ids, attention_mask, token_type_ids, label_id):
		self.input_ids = input_ids
		self.attention_mask = attention_mask
		self.token_type_ids = token_type_ids
		self.label_id = label_id

	def __repr__(self):
		return str(self.to_json_string())

	def to_dict(self):
		output = copy.deepcopy(self.__dict__)
		return output

	def to_json_string(self):
		return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class mydataProcessor(object):
	def __init__(self, args):
		self.args = args
		# self.relation_labels = get_label()
	@classmethod
	def _read_csv(cls, input_file, quotechar=None):
		with open(input_file, "r", encoding="utf-8") as f:
			reader = csv.reader(f, delimiter="\t")#, quotechar=quotechar)
			lines = []
			for line in reader:
				lines.append(line)
			return lines

	def _create_examples(self, lines, set_type):
		examples = []
		for (i, line) in enumerate(lines):
			guid = "%s-%s" % (set_type, i)
			text_a = line[0]
			text_b = line[1]
			#print(line[0],'--------',line[1],'--------',line[2])
			# label = self.relation_labels.index(line[2])
			label=float((line[2].split(":")[1]).strip(" "))
			if i % 1000 == 0:
				logger.info(line)
			examples.append(InputExample(guid=guid, text_a=text_a,text_b=text_b, label=label))
		return examples

	def get_examples(self, mode):
		file_to_read = None
		if mode == "train":
			file_to_read = self.args.train_file
		elif mode == "dev":
			file_to_read = self.args.dev_file
		elif mode == "test":
			file_to_read = self.args.test_file

		logger.info("LOOKING AT {}".format(os.path.join(self.args.data_dir, file_to_read)))
		return self._create_examples(self._read_csv(os.path.join(self.args.data_dir, file_to_read)), mode)

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()

def convert_examples_to_features(examples,max_seq_len,tokenizer,cls_token="[CLS]",cls_token_segment_id=0,sep_token="[SEP]",pad_token=0,pad_token_segment_id=0,add_sep_token=False,mask_padding_with_zero=True,):
	features = []
	count=0
	for (ex_index, example) in enumerate(examples):
		count+=1
		#print(count)
		if ex_index % 5000 == 0:
			logger.info("Writing example %d of %d" % (ex_index, len(examples)))

		tokens_a = tokenizer.tokenize(example.text_a)
		tokens_b = tokenizer.tokenize(example.text_b)

		# Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
		if add_sep_token:
			special_tokens_count = 3
		else:
			special_tokens_count = 2
		_truncate_seq_pair(tokens_a, tokens_b, max_seq_len - special_tokens_count)


		tokens = []
		token_type_ids = []
		tokens.append("[CLS]")
		token_type_ids.append(0)
		for token in tokens_a:
			tokens.append(token)
			token_type_ids.append(0)
		tokens.append("[SEP]")
		token_type_ids.append(0)

		for token in tokens_b:
			tokens.append(token)
			token_type_ids.append(1)
		if add_sep_token:
			tokens.append("[SEP]")
			token_type_ids.append(1)

		input_ids = tokenizer.convert_tokens_to_ids(tokens)
		attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

		# Zero-pad up to the sequence length.
		while len(input_ids) < max_seq_len:
			input_ids.append(0)
			attention_mask.append(0)
			token_type_ids.append(0)

		assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
		assert (len(attention_mask) == max_seq_len), "Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_len)
		assert (len(token_type_ids) == max_seq_len), "Error with token type length {} vs {}".format(len(token_type_ids), max_seq_len)

		# print(example.label)
		label_id = example.label

		if ex_index < 5:
			logger.info("*** Example ***")
			logger.info("guid: %s" % example.guid)
			logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
			logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
			logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
			logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
			logger.info("label: %s (id = %d)" % (example.label, label_id))

		features.append(InputFeatures(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids,label_id=label_id,))
	return features


def load_and_cache_examples(args,tokenizer,mode):
	processor = mydataProcessor(args)
	logger.info("Creating features from dataset file at %s", args.data_dir)
	if mode == "train":
		examples = processor.get_examples("train")
	elif mode == "dev":
		examples = processor.get_examples("dev")
	elif mode == "test":
		examples = processor.get_examples("test")
	else:
		raise Exception("For mode, Only train, dev, test is available")


	features = convert_examples_to_features(examples, args.max_seq_len, tokenizer, add_sep_token=args.add_sep_token)

	all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
	all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
	all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)

	all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

	dataset = TensorDataset(all_input_ids,all_attention_mask,all_token_type_ids,all_label_ids,)
	return dataset

MODEL_CLASSES = {"bert": (BertConfig, NoveltyClassifier, AutoTokenizer),}


MODEL_PATH_MAP = {"bert": "bert-base-uncased",}



# def get_label():
# 	return ["No", "Yes"]


def load_tokenizer(args):
	print(args.model_type)
	tokenizer = MODEL_CLASSES[args.model_type][2].from_pretrained(args.model_name_or_path)
	return tokenizer


def write_prediction(args, output_file, preds,orinal_label):
	# relation_labels = get_label()
	rd_test_data=open(os.path.join(args.data_dir,args.test_file),'r',encoding='utf-8')
	reader = csv.reader(rd_test_data, delimiter="\t", quotechar=None)
	lines = []
	for line in reader:
		lines.append(line)
	with open(output_file, "w", encoding="utf-8") as f:
		f.write("{}\t{}\t{}\n".format("Original","Paraphrase","Class"))
		for text_a, pred,actual in zip(lines,preds,orinal_label):
			f.write("{}\t{}\t{}\n".format(text_a[0], pred,actual))


def init_logger():
	logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",datefmt="%m/%d/%Y %H:%M:%S",level=logging.INFO,)


def set_seed(args):
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)


def compute_metrics(preds, labels):
	assert len(preds) == len(labels)
	return acc_and_f1(preds, labels)


def simple_accuracy(preds, labels):
	return (preds == labels).mean()


def acc_and_f1(preds, labels, average="macro"):
	acc = simple_accuracy(preds, labels)
	report = classification_report(labels, preds, digits=4)
	return {"acc": acc,"f1": report,}

class Trainer(object):
	def __init__(self, args, train_dataset=None, dev_dataset=None, test_dataset=None):
		self.args = args
		self.train_dataset = train_dataset
		self.dev_dataset = dev_dataset
		self.test_dataset = test_dataset

		# self.label_lst = get_label()
		self.num_labels = 1

		self.config_class, self.model_class, _ = MODEL_CLASSES[args.model_type]
		self.config = self.config_class.from_pretrained(args.model_name_or_path,num_labels=self.num_labels,)
		self.model = self.model_class.from_pretrained(args.model_name_or_path, config=self.config, args=args)

		# GPU or CPU
		self.device = self.args.device
		self.model.to(self.device)

	def train(self):
		train_sampler = RandomSampler(self.train_dataset)
		train_dataloader = DataLoader(self.train_dataset,sampler=train_sampler,batch_size=self.args.train_batch_size,)

		if self.args.max_steps > 0:
			t_total = self.args.max_steps
			self.args.num_train_epochs = (self.args.max_steps // (len(train_dataloader)// self.args.gradient_accumulation_steps)+ 1)
		else:
			t_total = (len(train_dataloader)// self.args.gradient_accumulation_steps* self.args.num_train_epochs)

		no_decay = ["bias", "LayerNorm.weight"]
		optimizer_grouped_parameters = [
			{
				"params": [
					p
					for n, p in self.model.named_parameters()
					if not any(nd in n for nd in no_decay)
				],
				"weight_decay": self.args.weight_decay,
			},
			{
				"params": [
					p
					for n, p in self.model.named_parameters()
					if any(nd in n for nd in no_decay)
				],
				"weight_decay": 0.0,
			},
		]
		optimizer = AdamW(optimizer_grouped_parameters,lr=self.args.learning_rate,eps=self.args.adam_epsilon,)
		scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=self.args.warmup_steps,num_training_steps=t_total,)

		# Train!
		logger.info("***** Running training *****")
		logger.info("  Num examples = %d", len(self.train_dataset))
		logger.info("  Num Epochs = %d", self.args.num_train_epochs)
		logger.info("  Total train batch size = %d", self.args.train_batch_size)
		logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
		logger.info("  Total optimization steps = %d", t_total)
		logger.info("  Logging steps = %d", self.args.logging_steps)
		logger.info("  Save steps = %d", self.args.save_steps)

		global_step = 0
		tr_loss = 0.0
		self.model.zero_grad()
		best_mse_loss = 999
		train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")

		for _ in train_iterator:
			epoch_iterator = tqdm(train_dataloader, desc="Iteration")
			for step, batch in enumerate(epoch_iterator):
				self.model.train()
				batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU
				inputs = {"input_ids": batch[0],"attention_mask": batch[1],"token_type_ids": batch[2],"labels": batch[3],}
				outputs = self.model(**inputs)
				loss = outputs[0]

				if self.args.gradient_accumulation_steps > 1:
					loss = loss / self.args.gradient_accumulation_steps

				loss.backward()

				tr_loss += loss.item()
				if (step + 1) % self.args.gradient_accumulation_steps == 0:
					torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

					optimizer.step()
					scheduler.step()  # Update learning rate schedule
					self.model.zero_grad()
					global_step += 1

				if 0 < self.args.max_steps < global_step:
					epoch_iterator.close()
					break
			mse_loss= self.evaluate("dev")
			# print(valid_f1,best_val_f1)
			if mse_loss < best_mse_loss:
				logger.info("Best validation score, saving best model")
				best_mse_loss=mse_loss
				self.save_model()
			else:
				logger.info("Not the best validation score, not overwriting best model")

			if 0 < self.args.max_steps < global_step:
				train_iterator.close()
				break
		return global_step, tr_loss / global_step
	def evaluate(self, mode):
		if mode == "test":
			dataset = self.test_dataset
		elif mode == "dev":
			dataset = self.dev_dataset
		else:
			raise Exception("Only dev and test dataset available")

		eval_sampler = SequentialSampler(dataset)
		eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

		# Eval!
		logger.info("***** Running evaluation on %s dataset *****", mode)
		logger.info("  Num examples = %d", len(dataset))
		logger.info("  Batch size = %d", self.args.eval_batch_size)
		eval_loss = 0.0
		nb_eval_steps = 0
		preds = None
		out_label_ids = None

		self.model.eval()

		for batch in tqdm(eval_dataloader, desc="Evaluating"):
			batch = tuple(t.to(self.device) for t in batch)
			with torch.no_grad():
				inputs = {"input_ids": batch[0],"attention_mask": batch[1],"token_type_ids": batch[2],"labels": batch[3],}
				outputs = self.model(**inputs)
				tmp_eval_loss, logits = outputs[:2]
				eval_loss += tmp_eval_loss.mean().item()
			nb_eval_steps += 1

			if preds is None:
				preds = logits.detach().cpu().numpy()
				out_label_ids = inputs["labels"].detach().cpu().numpy()
			else:
				preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
				out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

		eval_loss = eval_loss / nb_eval_steps
		results = {"loss": eval_loss}
		# preds = np.argmax(preds, axis=1)

		mse_loss = mean_squared_error(out_label_ids,preds)
		print(mse_loss)
		actual = []
		predicted = []
		if mode =='test':
			#------------------------------------------------------------------------------------------
			#pcc = pearson_cf(out_label_ids, preds)
			#print('Pearson Corelation co-efficient::', pcc)
			#corr, _ = pearsonr([out_label_ids], [preds])
			#print('Pearsons correlation: %.3f' % corr)
			for p in preds:
				predicted.append(p[0])
			for q in out_label_ids:
				actual.append(q)
			
			print(preds)
			print(out_label_ids)

			write_prediction(self.args, os.path.join(self.args.eval_dir, "proposed_answers_Train_Test_13_02_2021_run_"+fold+".txt"), preds,out_label_ids)
			#calculating accuracy manually

			res=open(os.path.join(self.args.eval_dir, "Results_Train_Test_13_02_2021_run_"+fold+".txt"),'w',encoding='utf-8')
			# count=0
			# count_p=0
			# count_n=0
			# i=0

			# ori_p=0
			# ori_n=0

			# while(i<len(out_label_ids)):
			# 	if(out_label_ids[i]==1):
			# 		ori_p+=1
			# 	elif(out_label_ids[i]==0):
			# 		ori_n+=1
			# 	if(preds[i]==out_label_ids[i]):
			# 		count+=1
			# 		if(out_label_ids[i]==1):
			# 			count_p+=1
			# 		elif(out_label_ids[i]==0):
			# 			count_n+=1
			# 	i+=1

			# res.write("Number of YES correctly predicted = "+str(count_p)+ " out of "+str(ori_p)+" YES sentences\n")
			# res.write("Number of NO correctly predicted = "+str(count_n)+ " out of "+str(ori_n)+" NO sentences\n")

			# res.write('\n\n\n')

			res.write("***********************************\n")
			res.write("BERT_TAP2.0_Novelty Detection"+"\n")
			res.write("***********************************\n")
			res.write(str(mse_loss)+'\n')
			# res.write("accuracy : " + str(result['acc']))
			#----------------------------------------------------------------------------------------------
		print(len(actual))
		print(len(predicted))

		data_dict ={'Actual':actual,'Predicted':predicted}
		df=pd.DataFrame.from_dict(data_dict)
		df.to_csv('TAP2-output_5.csv',index=True,sep='\t')

		#results.update(result)
		logger.info("***** Eval results *****")
		# for key in sorted(results.keys()):
		# 	print(key,results[key])
		# 	#logger.info("  {} = {:.4f}".format(key, results[key]))
		# # report = classification_report(out_label_ids, preds, digits=4)
		# # f1 = f1_score(out_label_ids, preds)
		# # print(report)
		return mse_loss

	def save_model(self):
		model_to_save = (self.model.module if hasattr(self.model, "module") else self.model)
		model_to_save.save_pretrained(self.args.model_dir)
		torch.save(self.args, os.path.join(self.args.model_dir, "training_args.bin"))
		logger.info("Saving model checkpoint to %s", self.args.model_dir)

	def load_model(self):
		self.args = torch.load(os.path.join(self.args.model_dir, "training_args.bin"))
		self.config = self.config_class.from_pretrained(self.args.model_dir)
		self.model = self.model_class.from_pretrained(self.args.model_dir, config=self.config, args=self.args)
		self.model.to(self.device)
		logger.info("***** Model Loaded *****")

def main(args):
	init_logger()
	set_seed(args)
	tokenizer = load_tokenizer(args)

	train_dataset = load_and_cache_examples(args, tokenizer, mode="train")
	dev_dataset = load_and_cache_examples(args, tokenizer, mode="dev")
	test_dataset = load_and_cache_examples(args, tokenizer, mode="test")

	trainer = Trainer(args, train_dataset=train_dataset, test_dataset=test_dataset, dev_dataset=dev_dataset)

	if args.do_train:
		trainer.train()

	if args.do_eval:
		trainer.load_model()
		trainer.evaluate("test")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_dir",default="Data/",type=str,help="The input data dir. Should contain the .tsv files (or other data files) for the task.",)
	parser.add_argument("--model_dir", default='Model1/', type=str, help="Path to model")
	parser.add_argument("--eval_dir",default='Result/',type=str,help="Evaluation script, result directory",)
	parser.add_argument("--train_file", default="Train_TAP2.0_Fold_"+fold+".csv", type=str, help="Train file")
	parser.add_argument("--dev_file", default="Dev_TAP2.0_Fold_"+fold+".csv", type=str, help="Dev file")
	parser.add_argument("--test_file", default="Test_TAP2.0_Fold_"+fold+".csv", type=str, help="Test file")
	parser.add_argument("--model_type",default="bert",type=str,help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),)
	parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
	parser.add_argument("--train_batch_size", default=4, type=int, help="Batch size for training.")
	parser.add_argument("--eval_batch_size", default=32, type=int, help="Batch size for evaluation.")
	parser.add_argument("--max_seq_len",default=512,type=int,help="The maximum total input sequence length after tokenization.",)
	parser.add_argument("--learning_rate",default=2e-5,type=float,help="The initial learning rate for Adam.",)
	parser.add_argument("--num_train_epochs",default=10.0,type=float,help="Total number of training epochs to perform.",)
	parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
	parser.add_argument("--gradient_accumulation_steps",type=int,default=1,help="Number of updates steps to accumulate before performing a backward/update pass.",)
	parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
	parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
	parser.add_argument("--max_steps",default=-1,type=int,help="If > 0: set total number of training steps to perform. Override num_train_epochs.",)
	parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
	parser.add_argument("--dropout_rate",default=0.1,type=float,help="Dropout for fully-connected layers",)
	parser.add_argument("--logging_steps", type=int, default=250, help="Log every X updates steps.")
	parser.add_argument("--save_steps",type=int,default=25,help="Save checkpoint every X updates steps.",)
	parser.add_argument("--do_train", action="store_true",default=True,help="Whether to run training.")
	parser.add_argument("--do_eval", action="store_true", default=True,help="Whether to run eval on the test set.")
	parser.add_argument("--add_sep_token",action="store_true",default=True,help="Add [SEP] token at the end of the sentence",)
	parser.add_argument("--device", default = torch.device("cuda:6"))
	args = parser.parse_args()
	args.model_name_or_path = MODEL_PATH_MAP[args.model_type]
	main(args)
