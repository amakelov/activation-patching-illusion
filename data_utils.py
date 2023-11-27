from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
from common_imports import *

def is_single_token(s: str, model: HookedTransformer) -> bool:
    """
    Check if a string is a single token in the vocabulary of a model.
    """
    try:
        model.to_single_token(s)
        return True
    except Exception as e:
        return False

ROOT = Path(__file__).parent
NAMES_PATH = ROOT / "data" / "names.json"
OBJECTS_PATH = ROOT / "data" / "objects.json"
PLACES_PATH = ROOT / "data" / "places.json"
PREFIXES_PATH = ROOT / "data" / "prefixes.json"
TEMPLATES_PATH = ROOT / "data" / "templates.json"

NAMES = json.load(open(NAMES_PATH))
OBJECTS = json.load(open(OBJECTS_PATH))
PLACES = json.load(open(PLACES_PATH))
PREFIXES = json.load(open(PREFIXES_PATH))
TEMPLATES = json.load(open(TEMPLATES_PATH))

class Prompt:
    """
    Represent a general ABC prompt using a template, and operations on it that
    are useful for generating datasets.
    """

    def __init__(
        self,
        names: Tuple[str, str, str],
        prefix: str,
        template: str,
        obj: str,
        place: str,
    ):
        self.names = names
        self.prefix = prefix
        self.template = template
        self.obj = obj
        self.place = place
        if self.is_ioi:
            self.s_name = self.names[2] # subject always appears in third position
            self.io_name = [x for x in self.names[:2] if x != self.s_name][0]
            self.s1_pos = self.names[:2].index(self.s_name)
            self.io_pos = self.names[:2].index(self.io_name)
            self.s2_pos = 2
        else:
            self.io_name = None
            self.s_name = None

    @property
    def is_ioi(self) -> bool:
        return self.names[2] in self.names[:2] and len(set(self.names)) == 2

    def __repr__(self) -> str:
        return f"<===PROMPT=== {self.sentence}>"

    @property
    def sentence(self) -> str:
        return self.prefix + self.template.format(
            name_A=self.names[0],
            name_B=self.names[1],
            name_C=self.names[2],
            object=self.obj,
            place=self.place,
        )

    @staticmethod
    def canonicalize(things: Tuple[str, str, str]) -> Tuple[str, str, str]:
        # the unique elements of the tuple, in the order they appear
        ordered_uniques = list(OrderedDict.fromkeys(things).keys())
        canonical_elts = ['A', 'B', 'C']
        uniques_to_canonical = {x: y for x, y in zip(ordered_uniques, canonical_elts[:len(ordered_uniques)])}
        return tuple([uniques_to_canonical[x] for x in things])

    @staticmethod
    def matches_pattern(names: Tuple[str, str, str], pattern: str) -> bool:
        return Prompt.canonicalize(names) == Prompt.canonicalize(tuple(pattern))
    
    def resample_pattern(self, orig_pattern: str, new_pattern: str,
                         name_distribution: Sequence[str]) -> "Prompt":
        """
        Change the pattern of the prompt, while keeping the names that are
        mapped to the same symbols in the original and new patterns the same.

        Args:
            orig_pattern (str): _description_
            new_pattern (str): _description_
            name_distribution (Sequence[str]): _description_

        Example:
            prompt = train_distribution.sample_one(pattern='ABB')
            (prompt.sentence, 
            prompt.resample_pattern(orig_pattern='ABB', new_pattern='BAA', 
                                    name_distribution=train_distribution.names,).sentence,
            prompt.resample_pattern(orig_pattern='ABB', new_pattern='CDD', 
                                    name_distribution=train_distribution.names,).sentence,
            prompt.resample_pattern(orig_pattern='ABB', new_pattern='ACC', 
                                    name_distribution=train_distribution.names,).sentence,
        
        >>> ('Then, Olivia and Anna had a long and really crazy argument. Afterwards, Anna said to',
        >>> 'Then, Anna and Olivia had a long and really crazy argument. Afterwards, Olivia said to',
        >>> 'Then, Joe and Kelly had a long and really crazy argument. Afterwards, Kelly said to',
        >>> 'Then, Olivia and Carl had a long and really crazy argument. Afterwards, Carl said to')
        )
        """
        assert len(orig_pattern) == 3
        assert len(new_pattern) == 3
        assert len(set(orig_pattern)) == len(set(new_pattern)) == 2
        assert self.matches_pattern(names=self.names, pattern=orig_pattern)
        orig_to_name = {orig_pattern[i]: self.names[i] for i in range(3)}
        new_names = [None for _ in range(3)]
        new_pos_to_symbol = {}
        for i, symbol in enumerate(new_pattern):
            if symbol in orig_to_name.keys():
                new_names[i] = orig_to_name[symbol]
            else:
                new_pos_to_symbol[i] = symbol
        new_symbols = new_pos_to_symbol.values()
        if len(new_symbols) > 0:
            new_symbol_to_name = {}
            # must sample some *new* names
            available_names = [x for x in name_distribution if x not in self.names]
            for symbol in new_symbols:
                new_symbol_to_name[symbol] = random.choice(available_names)
                available_names.remove(new_symbol_to_name[symbol])
            # populate new_names with new symbols
            for i, symbol in new_pos_to_symbol.items():
                new_names[i] = new_symbol_to_name[symbol]
        return Prompt(
            names=tuple(new_names),
            template=self.template,
            obj=self.obj,
            place=self.place,
            prefix=self.prefix,
        )

def load_data(data: Union[List[str], str, Path]) -> List[str]:
    if isinstance(data, (str, Path)):
        with open(data) as f:
            data: List[str] = json.load(f)
    return data


class PromptDataset(Dataset):
    def __init__(self, prompts: List[Prompt], model: HookedTransformer):
        assert len(prompts) > 0
        self.prompts: Sequence[Prompt] = np.array(prompts)
        self.model = model
        ls = self.lengths
        if not all(x == ls[0] for x in ls):
            raise ValueError("Prompts must all have the same length")

    def __getitem__(self, idx: Union[int, Sequence, slice]) -> "PromptDataset":
        if isinstance(idx, int):
            prompts = [self.prompts[idx]]
        else:
            prompts = self.prompts[idx]
            if isinstance(prompts, Prompt):
                prompts = [prompts]
        assert all(isinstance(x, Prompt) for x in prompts)
        return PromptDataset(prompts=prompts, model=self.model)

    def __len__(self) -> int:
        return len(self.prompts)

    def __repr__(self) -> str:
        return f"{[x for x in self.prompts]}"

    def __add__(self, other: "PromptDataset") -> "PromptDataset":
        return PromptDataset(
            prompts=list(self.prompts) + list(other.prompts), model=self.model
        )

    @property
    def lengths(self) -> List[int]:
        return [self.model.to_tokens(x.sentence).shape[1] for x in self.prompts]

    @property
    def tokens(self) -> Tensor:
        return self.model.to_tokens([x.sentence for x in self.prompts])

    @property
    def io_tokens(self) -> Tensor:
        return torch.tensor(
            [self.model.to_single_token(f" {x.io_name}") for x in self.prompts]
        )

    @property
    def s_tokens(self) -> Tensor:
        return torch.tensor(
            [self.model.to_single_token(f" {x.s_name}") for x in self.prompts]
        )

    @property
    def answer_tokens(self) -> JaxFloat[Tensor, "batch 2"]:
        # return a tensor with two columns: self.io_tokens and self.s_tokens
        return torch.tensor(
            [
                [
                    self.model.to_single_token(f" {x.io_name}"),
                    self.model.to_single_token(f" {x.s_name}"),
                ]
                for x in self.prompts
            ]
        )



class PatchingDataset(Dataset):
    """
    Bundle together the data needed to *train* a (DAS or other) patching for a
    single causal variable (we can generalize this later if we need).

    All you need to do *trainable* patching is the base and source
    `PromptDataset`s, and the patched answer tokens of shape (batch, 2), where
        - the 1st column is the patched answer,
        - and the 2nd column is the other possible answer (useful for computing
        logit diffs).

    Since this dataset holds only the bare minimum necessary for patching, it
    decouples the kind of patching we do from the data representation, allowing
    us to treat data in the same way regardless of whether we're doing DAS or
    some other kind of patching.
    """

    def __init__(
        self,
        base: PromptDataset,
        source: PromptDataset,
        patched_answer_tokens: JaxFloat[Tensor, "batch 2"],
    ):
        assert len(base) == len(source)
        assert len(base) == len(patched_answer_tokens)
        self.base = base
        self.source = source
        self.patched_answer_tokens = patched_answer_tokens.long()

    def batches(
        self, batch_size: int, shuffle: bool = True
    ) -> Iterable["PatchingDataset"]:
        if shuffle:
            order = np.random.permutation(len(self))
        else:
            order = np.arange(len(self))
        for i in range(0, len(self), batch_size):
            yield self[order[i : i + batch_size]]

    def __getitem__(self, idx) -> "PatchingDataset":
        patched_answer_tokens = self.patched_answer_tokens[idx]
        if len(patched_answer_tokens.shape) == 1:
            patched_answer_tokens = patched_answer_tokens.unsqueeze(0)
        return PatchingDataset(
            base=self.base[idx],
            source=self.source[idx],
            patched_answer_tokens=patched_answer_tokens,
        )

    def __len__(self) -> int:
        return len(self.base)
    
    def __add__(self, other: "PatchingDataset") -> "PatchingDataset":
        return PatchingDataset(
            base=self.base + other.base,
            source=self.source + other.source,
            patched_answer_tokens=torch.cat([self.patched_answer_tokens, other.patched_answer_tokens], 
                                            dim=0),
        )



class PromptDistribution:
    """
    A class to represent a distribution over prompts.

    It uses a combination of names, places, objects, prefixes, and templates
    loaded from JSON files or provided lists.

    Each prompt is constructed using a selected template and a randomly selected
    name, object, and place.

    Attributes
    ----------
    prefix_len : int
        The length of the prefix to use when creating the prompts.
    """

    def __init__(
        self,
        names: Union[List[str], str, Path],
        places: Union[List[str], str, Path],
        objects: Union[List[str], str, Path],
        prefixes: Union[List[str], str, Path],
        templates: Union[List[str], str, Path],
        prefix_len: int = 2,
    ):
        self.prefix_len = prefix_len
        self.names = load_data(names)
        self.places = load_data(places)
        self.objects = load_data(objects)
        self.prefixes = load_data(prefixes)
        self.templates = load_data(templates)

    def sample_one(self,
                   pattern: str, 
                   ) -> Prompt:
        """
        Sample a single prompt from the distribution.
        """
        template = random.choice(self.templates)
        unique_ids = list(set(pattern))
        unique_names = random.sample(self.names, len(unique_ids))
        assert len(set(unique_names)) == len(unique_names)
        prompt_names = tuple([unique_names[unique_ids.index(i)] for i in pattern])
        obj = random.choice(self.objects)
        place = random.choice(self.places)
        prefix = self.prefixes[self.prefix_len]
        return Prompt(
            names=prompt_names, template=template, obj=obj, place=place, prefix=prefix
        )

    def sample_das(
        self,
        model: HookedTransformer,
        base_patterns: List[str],
        source_patterns: List[str],
        samples_per_combination: int,
        labels: Literal["position", "name"],
    ) -> PatchingDataset:
        """
        This samples a dataset of base and corrupted prompts for doing DAS on
        position or name subspaces.

        model : HookedTransformer
            The model that will be used to convert the prompts to tokens.
        samples_per_combination : int
            The number of samples to be generated for each combination of patterns.
        orig_patterns : List[str]
            A list of original patterns that will be used to create the prompts. For example ["ABB", "BAB"].
        corrupted_patterns : List[str]
            A list of corrupted patterns that will be used to create the corrupted prompts.
            Use same letters as in orig_patterns if you want to use the same names, objects, and places.
            Use different letters like ["CDD", "DCD"] if you want to use different names, objects, and places.
        labels : str
            The type of label for the task. Supports 'position' and 'name'.
            The label is the answer token that the model should predict if the position or name information is patched
            into activations during the forward pass of the clean prompt.
        """
        base_prompts: List[Prompt] = []
        source_prompts: List[Prompt] = []
        for orig_pattern in base_patterns:
            for corrupted_pattern in source_patterns:
                base_prompt_batch = [
                    self.sample_one(orig_pattern)
                    for _ in range(samples_per_combination)
                ]
                source_prompt_batch = [
                    p.resample_pattern(
                        name_distribution=self.names,
                        orig_pattern=orig_pattern,
                        new_pattern=corrupted_pattern,
                        ) for p in base_prompt_batch
                ]
                base_prompts.extend(base_prompt_batch)
                source_prompts.extend(source_prompt_batch)

        # if DAS should find the position subspace
        if labels == "position":
            patched_answer_names = [] # list of (correct, incorrect) name pairs
            for base_prompt, source_prompt in zip(base_prompts, source_prompts):
                if base_prompt.s1_pos == source_prompt.s1_pos:
                    patched_answer_names.append(
                        (base_prompt.io_name, base_prompt.s_name)
                    )
                else:
                    patched_answer_names.append(
                        (base_prompt.s_name, base_prompt.io_name)
                    )
        else:
            raise NotImplementedError(f"Labels {labels} not implemented")

        clean_dataset = PromptDataset(base_prompts, model)
        corrupted_dataset = PromptDataset(source_prompts, model)
        patched_answer_tokens = torch.Tensor(
            [[model.to_single_token(f" {x}") for x in y] # prepend space for each name
             for y in patched_answer_names]
        )
        return PatchingDataset(
            base=clean_dataset,
            source=corrupted_dataset,
            patched_answer_tokens=patched_answer_tokens,
        )


train_distribution = PromptDistribution(
    names=NAMES[:len(NAMES) // 2],
    objects=OBJECTS[:len(OBJECTS) // 2],
    places=PLACES[:len(PLACES) // 2],
    prefix_len=2,
    prefixes=PREFIXES,
    templates=TEMPLATES[:2]
)

test_distribution = PromptDistribution(
    names=NAMES[len(NAMES) // 2:],
    objects=OBJECTS[len(OBJECTS) // 2:],
    places=PLACES[len(PLACES) // 2:],
    prefix_len=2,
    prefixes=PREFIXES,
    templates=TEMPLATES[2:]
)