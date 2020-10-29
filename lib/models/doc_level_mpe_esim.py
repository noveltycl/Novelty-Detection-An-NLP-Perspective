from typing import Dict, Optional, List, Any
import copy
import re

from overrides import overrides
import torch

from allennlp.common import Params
from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, InputVariationalDropout, Seq2VecEncoder
from allennlp.modules.matrix_attention.legacy_matrix_attention import LegacyMatrixAttention
from allennlp.modules import Seq2SeqEncoder, SimilarityFunction, TextFieldEmbedder
from allennlp.modules.time_distributed import TimeDistributed
from allennlp.modules.similarity_functions.dot_product import DotProductSimilarity
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, masked_softmax, weighted_sum, replace_masked_values, get_mask_from_sequence_lengths
from allennlp.training.metrics import BooleanAccuracy

from lib.modules import CoverageLoss
from lib.nn.util import unbind_tensor_dict
from lib.models.mpe_esim import mpeEsim


@Model.register("doc_level_mpe_esim")
class DocLevelmpeEsim(mpeEsim):

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 projection_feedforward: FeedForward,
                 inference_encoder: Seq2SeqEncoder,
                 output_feedforward: FeedForward,
                 output_logit: FeedForward,
                 final_feedforward: FeedForward,
                 coverage_loss: CoverageLoss,
                 similarity_function: SimilarityFunction = DotProductSimilarity(),
                 dropout: float = 0.5,
                 contextualize_pair_comparators: bool = False,
                 pair_context_encoder: Seq2SeqEncoder = None,
                 pair_feedforward: FeedForward = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
    	# Need to send it verbatim because otherwise FromParams doesn't work appropriately.
        super().__init__(vocab=vocab,
                         text_field_embedder=text_field_embedder,
                         encoder=encoder,
                         similarity_function=similarity_function,
                         projection_feedforward=projection_feedforward,
                         inference_encoder=inference_encoder,
                         output_feedforward=output_feedforward,
                         output_logit=output_logit,
                         final_feedforward=final_feedforward,
                         contextualize_pair_comparators=contextualize_pair_comparators,
                         coverage_loss=coverage_loss,
                         pair_context_encoder=pair_context_encoder,
                         pair_feedforward=pair_feedforward,
                         dropout=dropout,
                         initializer=initializer,
                         regularizer=regularizer)
        self._answer_loss = torch.nn.BCELoss()
        self.max_sent_count = 120
        self.fc1 = torch.nn.Linear(self.max_sent_count,10)
        self.fc2 = torch.nn.Linear(10,5)
        self.fc3 = torch.nn.Linear(5,1) 
        self.out_sigmoid = torch.nn.Sigmoid()

        self._accuracy = BooleanAccuracy()

    @overrides
    def forward(self,  # type: ignore
                premises: Dict[str, torch.LongTensor],
                hypotheses: Dict[str, torch.LongTensor],
                paragraph: Dict[str, torch.LongTensor],
                answer_index: torch.LongTensor = None,
                relevance_presence_mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        hypothesis_list = unbind_tensor_dict(hypotheses, dim=1)

        label_logits = []
        premises_attentions = []
        premises_aggregation_attentions = []
        #coverage_losses = []
        for hypothesis in hypothesis_list: # single hypothesis even to the parent class
            #print("super().forward",len(premises), len(hypothesis), len(paragraph))
            output_dict = super().forward(premises=premises, hypothesis=hypothesis, paragraph=paragraph) #paragraph?
            individual_logit = output_dict["label_logits"][:, self._label2idx["entailment"]] # only useful key
            label_logits.append(individual_logit)
            #
            premises_attention = output_dict.get("premises_attention", None)
            premises_attentions.append(premises_attention)
            premises_aggregation_attention = output_dict.get("premises_aggregation_attention", None)
            premises_aggregation_attentions.append(premises_aggregation_attention)
            #if relevance_presence_mask is not None:
                #coverage_loss = output_dict["coverage_loss"]
                #coverage_losses.append(coverage_loss)
            del output_dict,individual_logit,premises_attention,premises_aggregation_attention

        label_logits = torch.stack(label_logits, dim=-1)
        premises_attentions = torch.stack(premises_attentions, dim=1)
        premises_aggregation_attentions = torch.stack(premises_aggregation_attentions, dim=1)
        #if relevance_presence_mask is not None:
            #coverage_losses = torch.stack(coverage_losses, dim=0)
        

        label_probs = torch.nn.functional.softmax(label_logits, dim=-1)
        # @todo:  Check covaraince of label_logits and label_probs
        if label_logits.shape[1] < self.max_sent_count:
            label_logits = torch.nn.functional.pad(input=label_logits, pad=(0, self.max_sent_count-label_logits.shape[1], 0, 0), mode='constant', value=0)
        
        single_output_logit = self.fc3(self.fc2(self.fc1(label_logits)))
        sigmoid_output = self.out_sigmoid(single_output_logit)
        #import pdb; pdb.set_trace()

        output_dict = {"label_logits": single_output_logit,
                       "label_probs": sigmoid_output,
                       "premises_attentions": premises_attentions,
                       "premises_aggregation_attentions": premises_aggregation_attentions}
        
        if answer_index is not None:
            #print("_answer_loss",single_output_logit, answer_index)
            cudadevice = single_output_logit.device # torch.device('cuda:'+ str(single_output_logit.get_device()))
            temp_tensor = torch.tensor([[k] for k in answer_index]).to(cudadevice)
            sgd = torch.nn.Sigmoid()
            loss = self._answer_loss(sgd(single_output_logit), sgd(temp_tensor.float()))
            output_dict["loss"] = loss
            output_dict["novelty"] = (single_output_logit>0.5)
            temp_tensor = torch.tensor([[k] for k in answer_index])
            #print("_answer_loss",single_output_logit, temp_tensor)
            self._accuracy(single_output_logit>0.5, temp_tensor.byte())
            del temp_tensor, loss, cudadevice

            #self._accuracy(single_output_logit>0.5, answer_index)
        del label_logits, label_probs, hypothesis_list, 
        # if answer_index is not None:
            # answer_loss
            # loss = self._answer_loss(label_logits, answer_index)
            # coverage loss
            # if relevance_presence_mask is not None:
            #     loss += coverage_losses.mean()
            # output_dict["loss"] = loss

            # self._accuracy(label_logits, answer_index)

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        accuracy_metric = self._accuracy.get_metric(reset)
        return {'accuracy': accuracy_metric}
