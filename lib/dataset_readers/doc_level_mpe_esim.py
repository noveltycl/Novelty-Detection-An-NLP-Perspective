import logging
from typing import Dict, List
from overrides import overrides
import json

import numpy as np
import tqdm

from allennlp.common import Params
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, ListField, TextField, ArrayField, LabelField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer, Token
from allennlp.common.checks import ConfigurationError

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("doc_level_mpe_esim")
class DocLevelmpeEsimReader(DatasetReader):

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 premise_max_tokens: int = 200, # do dataset statisticscs
                 hypothesis_max_tokens: int = 200,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._premise_max_tokens = premise_max_tokens
        self._hypothesis_max_tokens = hypothesis_max_tokens
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str):
        file_path = cached_path(file_path)
        with open(file_path, 'r') as entailment_file:
            logger.info("Reading entailment instances from jsonl dataset at: %s", file_path)
            for line in entailment_file:
                if line.strip():
                    instances_json = json.loads(line.strip())
                    premises = instances_json["premises"][:10]
                    hypotheses = instances_json["hypothesis"] # instances_json["hypotheses"]
                    entailments = instances_json.get("entailments", None)
                    if entailments is None or len(entailments)<1:
                        answer_index = None
                    else:
                        answer_index = entailments[0]
                    relevant_sentence_idxs = instances_json.get("relevant_sentence_idxs", None)
                    k = self.text_to_instance(premises, hypotheses, answer_index, relevant_sentence_idxs)
                    if k is not None:
                        yield k #self.text_to_instance(premises, hypotheses, answer_index, relevant_sentence_idxs)

    @overrides
    def text_to_instance(self, # pylint: disable=arguments-differ
                         premises: List[str],
                         hypotheses: List[str],
                         answer_index: int = None,
                         relevant_sentence_idxs: List[int] = None) -> Instance:
        fields = {}
        premises_tokens = [self._tokenizer.tokenize(premise)[-self._premise_max_tokens:]
                           for premise in premises]
        hypotheses_tokens = [self._tokenizer.tokenize(hypothesis)[-self._hypothesis_max_tokens:]
                             for hypothesis in hypotheses]
        if premises:
            premises_text_fields = [TextField(premise_tokens, self._token_indexers)
                                    for premise_tokens in premises_tokens]
            premises_field = ListField(premises_text_fields)
        else:
            empty_stub = ListField([TextField([Token('dummy')], self._token_indexers)])
            premises_field = empty_stub.empty_field()
        fields['premises'] = premises_field

        hypotheses_text_fields = [TextField(hypothesis_tokens, self._token_indexers)
                                for hypothesis_tokens in hypotheses_tokens]
        hypotheses_field = ListField(hypotheses_text_fields)
        fields['hypotheses'] = hypotheses_field

        # If sentence relevance is available
        if relevant_sentence_idxs is not None:
            relevance_presence_mask = np.zeros(len(premises))
            for idx in relevant_sentence_idxs:
                relevance_presence_mask[idx] = 1
            fields['relevance_presence_mask'] = ArrayField(np.array(relevance_presence_mask))

        # If entailment labels are available
        if answer_index is not None:
            # if answer_index not in range(0, len(hypotheses)):
            #     raise ConfigurationError("Provided label must be in 0 to {}".format(len(hypotheses)))
            fields['answer_index'] = ArrayField(np.array(answer_index), padding_value=-1, dtype=np.long)

        paragraph_tokens = [token for premise_tokens in premises_tokens for token in premise_tokens]
        #print(len(paragraph_tokens))
        if (len(paragraph_tokens)==0):
            return None
        paragraph_text_field = TextField(paragraph_tokens, self._token_indexers)

        fields['paragraph'] = paragraph_text_field
        return Instance(fields)
