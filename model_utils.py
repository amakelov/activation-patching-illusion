from common_imports import *
from torch.nn.utils.parametrizations import orthogonal
from torch.nn import functional as F
from data_utils import *
from transformer_lens import utils
from transformer_lens.hook_points import HookPoint
from fancy_einsum import einsum
from batched_decorator import batched

def get_model(model_name: str = "gpt2-small") -> HookedTransformer:
    model = HookedTransformer.from_pretrained(
        model_name=model_name,
        center_unembed=True,
        center_writing_weights=True,
        fold_ln=True,
        refactor_factored_attn_matrices=True,
    )
    model.requires_grad_(False)
    return model

class Node:
    """
    Mostly a copy of the one in path_patching.py, we'll see if it diverges
    """

    def __init__(
        self,
        component_name: Literal[
            "z",
            "attn_out",
            "pre",
            "post",
            "mlp_out",
            "resid_pre",
            "resid_post",
            "resid_mid",
            "q",
            "k",
            "v",
            "pattern",
            "attn_scores",
            "result",
            "q_input",
            "k_input",
            "v_input",
            'scale_ln1',
            'scale_ln2',
            'scale_final',
        ],
        layer: Optional[int] = None,
        head: Optional[int] = None,
        neuron: Optional[int] = None,
        seq_pos: Optional[int] = None,
    ):
        assert isinstance(component_name, str)
        self.component_name = component_name
        if layer is not None:
            assert isinstance(layer, int)
        self.layer = layer
        if head is not None:
            assert isinstance(head, int)
        self.head = head
        if neuron is not None:
            assert isinstance(neuron, int)
        self.neuron = neuron
        if seq_pos is not None:
            assert isinstance(seq_pos, int)
        self.seq_pos = seq_pos

    @property
    def activation_name(self) -> str:
        if self.component_name == 'scale_ln1':
            return utils.get_act_name('scale', layer=self.layer, layer_type='ln1')
        elif self.component_name == 'scale_ln2':
            return utils.get_act_name('scale', layer=self.layer, layer_type='ln2')
        elif self.component_name == 'scale_final':
             return utils.get_act_name('scale', layer=None)
        else:
            return utils.get_act_name(self.component_name, layer=self.layer)

    @property
    def shape_type(self) -> List[str]:
        """
        List of the meaning of each dimension of the full activation for this
        node (i.e., what you'd get if you did `cache[self.activation_name]`).
        
        This is just for reference
        """
        if self.component_name in [
            "resid_pre",
            "resid_post",
            "resid_mid",
            "q_input",
            "k_input",
            "v_input",
        ]:
            return ["batch", "seq", "d_model"]
        elif self.component_name == 'pattern':
            return ["batch", "head", "query_pos", "key_pos"]
        elif self.component_name in ["q", "k", "v", "z"]:
            return ["batch", "seq", "head", "d_head"]
        elif self.component_name in ["result"]:
            return ["batch", "seq", "head", "d_model"]
        elif self.component_name == 'scale':
            return ['batch', 'seq']
        elif self.component_name == 'post':
            return ['batch', 'seq', 'd_mlp']
        else:
            raise NotImplementedError

    @property
    def idx(self) -> Tuple[Union[int, slice, None], ...]:
        """
        Index into the full activation to restrict to layer / head / neuron /
        seq_pos
        """
        if self.neuron is not None:
            raise NotImplementedError
        elif self.component_name in ['pattern', 'attn_scores']:
            assert self.head is not None
            return tuple([slice(None), self.head, slice(None), slice(None)])
        elif self.component_name in ["q", "k", "v", "z", "result"]:
            assert self.head is not None, "head must be specified for this component"
            if self.seq_pos is not None:
                return tuple([slice(None), self.seq_pos, self.head, slice(None)])
            else:
                return tuple([slice(None), slice(None), self.head, slice(None)])
        elif self.component_name == 'scale':
            return tuple([slice(None), slice(None)])
        elif self.component_name == 'post':
            if self.seq_pos is not None:
                return tuple([slice(None), self.seq_pos, slice(None)])
            else:
                return tuple([slice(None), slice(None), slice(None)])
        else:
            if self.seq_pos is not None:
                return tuple([slice(None), self.seq_pos, slice(None)])
            else:
                return tuple([slice(None), slice(None), slice(None)])
    
    @property
    def names_filter(self) -> Callable:
        return lambda x: x in [self.activation_name]
    
    @staticmethod
    def get_names_filter(nodes: List['Node']) -> Callable:
        return lambda x: any(node.names_filter(x) for node in nodes)

    @property
    def needs_head_results(self) -> bool:
        return self.component_name in ['result']
    
    def get_value(self, cache: ActivationCache) -> Tensor:
        return cache[self.activation_name][self.idx]
    
    def __repr__(self) -> str:
        properties = OrderedDict({
            "component_name": self.component_name,
            "layer": self.layer,
            "head": self.head,
            "neuron": self.neuron,
            "seq_pos": self.seq_pos,
        })
        properties = ", ".join(f"{k}={v}" for k, v in properties.items() if v is not None)
        return f"Node({properties})"
    


################################################################################
### patching utils
################################################################################
class PatchImplementation(ABC):
    """
    This is a class instead of a function b/c with a function it's hard to
    access the state of the patcher (e.g. the rotation, the direction, who knows
    what). This class is used to store the state of the patcher.
    """

    @abstractmethod
    def __call__(
        self,
        base_activation: Tensor,
        source_activation: Tensor,
    ) -> Tensor:
        """
        base_activation: the activation to patch
        source_activation: the activation to use for the patch
        """
        pass

    @abstractmethod
    def parameters(self) -> Iterable[Parameter]:
        """
        Parameters of the patching function
        """
        pass


class Full(PatchImplementation):
    def __init__(self):
        pass

    def __call__(self, base_activation: Tensor, source_activation: Tensor) -> Tensor:
        return source_activation

    def parameters(self) -> Iterable[Parameter]:
        return []


class RotationMatrix(nn.Module):
    """
    Parametrized rotation matrix that can be optimized
    """

    def __init__(self, n: int):
        super().__init__()
        self.R = orthogonal(nn.Linear(n, n, bias=False))
        self.n = n

    def forward(self, x):
        # x is of shape [batch, activation_dim]
        return einsum(
            "batch activation_dim, activation_dim rotated_dim -> batch rotated_dim",
            x,
            self.R.weight,
        )

    def inverse(self, x):
        return einsum(
            "batch rotated_dim, rotated_dim activation_dim -> batch activation_dim",
            x,
            self.R.weight.T,
        )

    @staticmethod
    def load_rotation(path: Union[str, Path]) -> "RotationMatrix":
        data = torch.load(path)
        n = data["n"]
        state_dict = data["state_dict"]
        rotation = RotationMatrix(n=n)
        rotation.R.load_state_dict(state_dict)
        return rotation
    
    @staticmethod
    def load_rotation_old(path: Path, n: int) -> "RotationMatrix":
        sd = torch.load(path)
        rotation = RotationMatrix(n=n)
        rotation.R.load_state_dict(sd)
        return rotation
    
    @staticmethod
    def from_state_dict(sd: Dict[str, torch.Tensor], n: int) -> "RotationMatrix":
        rotation = RotationMatrix(n=n)
        rotation.R.load_state_dict(sd)
        return rotation

    def dump(self, path: str) -> None:
        data = {"n": self.n, "state_dict": self.R.state_dict()}
        torch.save(data, path)


class Rotation(PatchImplementation):
    """
    DAS patching a single variable
    """

    def __init__(self, rotation: RotationMatrix, dim: int):
        self.rotation = rotation
        self.dim = dim
        self.last_intermediates = {}

    def __call__(
        self,
        base_activation: JaxFloat[Tensor, "batch ..."],
        source_activation: JaxFloat[Tensor, "batch ..."],
    ) -> Tensor:
        base_rot = self.rotation(base_activation)
        source_rot = self.rotation(source_activation)
        patched_rot = torch.cat([source_rot[:, :self.dim], base_rot[:, self.dim:]], dim=1)
        return self.rotation.inverse(patched_rot)

    def parameters(self) -> Iterable[Parameter]:
        return self.rotation.parameters()

class DirectionPatch(PatchImplementation):
    """
    Patch values along a 1-dim subspace, i.e. a direction in activation
    space.

    Suppose we have vectors u (source), w (base) and a direction v. We want
    to change w to w' so that <w', v> = <u, v>, and <w', z> = <w, z> for all
    z orthogonal to v. We can do this by adding a multiple of v to w:
        w' = w + a * v

    <w', v> = <w, v> + a * <v, v> = <w, v> + a * ||v||_2^2
    We want this to be equal to <u, v>, so we solve for a:
        a = (<u, v> - <w, v>) / ||v||_2^2
    """

    def __init__(self, v: Tensor):
        self.v = v

    def __call__(self, base_activation: Tensor, source_activation: Tensor) -> Tensor:
        v = self.v
        assert base_activation.shape == source_activation.shape
        assert base_activation.shape[1:] == v.shape
        base_proj = einsum("batch ..., ... -> batch", base_activation, v)
        source_proj = einsum("batch ..., ... -> batch", source_activation, v)
        norm = v.norm()
        base_activation += einsum(
            "batch, ... -> batch ...", (source_proj - base_proj) / norm**2, v
        )
        return base_activation

    def parameters(self) -> Iterable[Parameter]:
        return []

class Patcher:
    """
    A location where to perform a patch (DAS or other). It is either
        - a single node (e.g. residual stream at a given layer and position), in
          which case it works like you'd expect
        - or several nodes (only head results at the same position are supported
        for this, in which case the patch is applied to the residual stream, as
        soon as all heads have been computed.

    It decouples the
    - "where": which components, which layer, which position (`nodes` argument)
    - "how": this is implemented in the `patch_fn` argument
    - "when": the `get_hook` returns a hook you can combine w/ other hooks and
    whatever to do more complex things (e.g. DAS + path patching)

    """

    def __init__(
        self,
        nodes: List[Node],
        patch_impl: PatchImplementation,
    ):
        """
        nodes: which activations to patch
        patch_fn: (base activation, source activation) -> patched activation
        """
        self.nodes = nodes
        self.patch_impl = patch_impl

        assert all(node.layer is not None for node in nodes)
        assert all(node.seq_pos is not None for node in nodes)

    @property
    def needs_head_results(self) -> bool:
        """
        Whether we need to call `compute_head_results` before patching
        """
        return len(self.nodes) > 1

    @property
    def names_filter(self) -> Callable:
        """
        Get a thing that can be used as the `names_filter` argument of
        `model.run_with_cache`. It filters the activations to only keep the
        ones needed for the patching (reducing memory usage).

        Returns:
            Callable: _description_
        """
        if len(self.nodes) == 1:
            return lambda x: x in [self.nodes[0].activation_name]
        else:
            act_names = [node.activation_name for node in self.nodes] + [
                self.target_node.activation_name
            ]
            # unfortunately, `compute_head_results` requires all the `z`
            # activations to be present; don't know how to get around it
            return lambda x: ("hook_z" in x) or x in act_names

    @property
    def target_node(self) -> Node:
        """
        The node at which we do the actual intervention. This is useful if we
        are patching head results, b/c then we patch the sum of heads in the
        residual stream.
        """
        if len(self.nodes) == 1:
            return self.nodes[0]
        else:
            if not all(x.component_name == "result" for x in self.nodes):
                raise NotImplementedError("Only head results are supported")
            max_layer = max(x.layer for x in self.nodes)
            return Node(
                component_name="resid_mid",
                layer=max_layer,
                seq_pos=self.nodes[0].seq_pos,
            )

    def patch_slice(
        self,
        base_slice: Tensor,
        source_slice: Tensor,
        cache_base: Optional[ActivationCache] = None,
        cache_source: Optional[ActivationCache] = None,
    ) -> Tensor:
        """
        This runs the actual patching on the *relevant slices* of the
        activations, i.e.  what you'd get when you restrict to proper seq_pos,
        head, etc.

        Returns the slice you should put back in the full activation.
        """
        if len(self.nodes) == 1:
            return self.patch_impl(base_activation=base_slice,
                                   source_activation=source_slice)
        else:
            # patch resid contributions
            assert cache_base is not None
            assert cache_source is not None
            idxs = [node.idx for node in self.nodes]
            summed_base = sum(
                [
                    cache_base[node.activation_name][idx]
                    for node, idx in zip(self.nodes, idxs)
                ]
            )
            summed_source = sum(
                [
                    cache_source[node.activation_name][idx]
                    for node, idx in zip(self.nodes, idxs)
                ]
            )
            patched = self.patch_impl(
                base_activation=summed_base, 
                source_activation=summed_source
                )
            return base_slice - summed_base + patched

    def get_hook(
        self,
        cache_base: Optional[ActivationCache],
        cache_source: ActivationCache,
        slice_method: str = 'mask'
    ) -> Tuple[str, Callable]:
        """
        Return a pair (activation_name, hook_fn) that can be used to perform the
        full patching.
        """

        def hook_fn(base_activation: Tensor, hook: HookPoint) -> Tensor:
            idx = self.target_node.idx
            activation_name = self.target_node.activation_name
            base_activation_slice = base_activation[idx]
            source_slice = cache_source[activation_name][idx]
            new_activation_slice = self.patch_slice(
                base_slice=base_activation_slice,
                source_slice=source_slice,
                cache_base=cache_base,
                cache_source=cache_source,
            )

            if slice_method == 'obvious':
                base_activation[idx] = new_activation_slice
                return base_activation
            elif slice_method == 'mask':
                # This is a very weird hack for the in-place backprop problem
                # some sanity checks for the shapes. The slice should be 2D with
                # the seq_pos being set to a constant value, and we insert this
                # in the 3D tensor of the base activation, where the middle dim
                # encodes the seq_pos
                assert len(new_activation_slice.shape) == 2
                assert len(base_activation.shape) == 3
                assert new_activation_slice.shape[0] == base_activation.shape[0]
                assert new_activation_slice.shape[1] == base_activation.shape[2]

                # Construct a boolean mask of the same shape as base_activation
                mask = torch.zeros_like(base_activation, dtype=torch.bool)
                mask[idx] = 1
                # Construct the new tensor using the mask
                base_activation_new = torch.where(mask, new_activation_slice.unsqueeze(1), base_activation)
                return base_activation_new

        return (self.target_node.activation_name, hook_fn)

    @batched(
        args=["P_base", "P_source", "answer_tokens_base", "answer_tokens_source", "patched_answer_tokens"],
        n_outputs=4,
        reducer="cat",
        shuffle=False,
    )
    def run_patching(
        self,
        model: HookedTransformer,
        P_base: JaxFloat[Tensor, "batch seq_len"],
        P_source: JaxFloat[Tensor, "batch seq_len"],
        answer_tokens_base: Tensor,
        answer_tokens_source: Tensor,
        patched_answer_tokens: Tensor,
        return_full_patched_logits: bool = False,
        batch_size: Optional[int] = None,
    ) -> Tuple[
        JaxFloat[Tensor, "batch 2"],
        JaxFloat[Tensor, "batch 2"],
        JaxFloat[Tensor, "batch 2"],
        Optional[JaxFloat[Tensor, "batch d_vocab"]],
    ]:
        """
        Return logits_patched, logits_base, logits_source.

        Will be run in batches if `batch_size` is specified.
        """
        assert len(P_base) == len(P_source)
        names_filter = self.names_filter
        logits_base, cache_base = model.run_with_cache(
            P_base, names_filter=names_filter
        )
        if self.needs_head_results:
            cache_base.compute_head_results()
        answer_logits_base = logits_base[:, -1, :].gather(dim=1, index=answer_tokens_base.cuda())
        logits_source, cache_source = model.run_with_cache(
            P_source, names_filter=names_filter
        )
        if self.needs_head_results:
            cache_source.compute_head_results()
        answer_logits_source = logits_source[:, -1, :].gather(dim=1, index=answer_tokens_source.cuda())

        hk = self.get_hook(cache_base=cache_base, cache_source=cache_source)
        model.reset_hooks()
        logits_patched = model.run_with_hooks(P_base, fwd_hooks=[hk])[:, -1, :]
        answer_logits_patched = logits_patched.gather(dim=1, index=patched_answer_tokens.cuda())

        if return_full_patched_logits:
            return answer_logits_patched, answer_logits_base, answer_logits_source, logits_patched
        else:
            return answer_logits_patched, answer_logits_base, answer_logits_source, None

    @batched(
        args=["X_base", "X_source"],
        n_outputs=1,
        reducer="cat",
        shuffle=False,
        verbose=True,
    )
    def get_patched_activation(
        self, model: HookedTransformer, 
        node: Node,
        X_base: Optional[Tensor] = None,
        X_source: Optional[Tensor] = None,
        cache_base: Optional[ActivationCache] = None,
        cache_source: Optional[ActivationCache] = None,
        batch_size: Optional[int] = None,
        ) -> Tensor:
        """
        Return the activation of the given node after patching with this
        patcher.
        """
        if cache_base is None:
            _, cache_base = model.run_with_cache(X_base, names_filter=self.names_filter)
        if cache_source is None:
            _, cache_source = model.run_with_cache(X_source, names_filter=self.names_filter)
        if self.needs_head_results:
            cache_base.compute_head_results()
            cache_source.compute_head_results()
        hk = self.get_hook(cache_base=cache_base, cache_source=cache_source)
        model.reset_hooks()
        model.add_hook(name=hk[0], hook=hk[1], dir='fwd', is_permanent=False)
        node_filter = node.names_filter
        _, cache_patched = model.run_with_cache(X_base, names_filter=node_filter)
        model.reset_hooks()
        return node.get_value(cache_patched)


################################################################################
### patch training
################################################################################
@batched(args=["dataset"], n_outputs=3, reducer="cat", shuffle=False, verbose=False)
def eval_patch_logitdiffs(
    model: HookedTransformer,
    dataset: PatchingDataset,
    patcher: Patcher,
    batch_size: Optional[int] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Given a patcher and a patching dataset, return the 
        - ld_base: base_logits(base_answer_tokens[:, 0]) -
          base_logits(base_answer_tokens[:, 1:])
        - ld_patched: patched_logits(base_answer_tokens[:, 0]) -
            patched_logits(base_answer_tokens[:, 1:])
        - logits_patched: the patched logits
    Args:
        model: the model to patch
        dataset: the patching dataset w/ base and source datasets
        patcher: the patcher to use
        batch_size: batch size to use for patching; if None, do it all at once
    """
    answer_logits_patched, answer_logits_base, _, _ = patcher.run_patching(
        model, dataset.base.tokens, dataset.source.tokens,
        answer_tokens_base=dataset.base.answer_tokens.cuda(),
        answer_tokens_source=dataset.source.answer_tokens.cuda(),
        patched_answer_tokens=dataset.patched_answer_tokens.cuda(),
        return_full_patched_logits=False,
    )

    ld_base = answer_logits_base[:, 0] - answer_logits_base[:, 1]
    ld_patched = answer_logits_patched[:, 0] - answer_logits_patched[:, 1]
    return ld_base, ld_patched, answer_logits_patched


def metrics_to_str(metrics: Dict[str, float]) -> str:
    """
    Convert a dictionary of metrics to a string of the form "k1=v1 k2=v2 ...".
    """
    return " ".join(f"{k}={v:.3f}" for k, v in metrics.items())

def get_acc_from_logits(answer_logits: JaxFloat[Tensor, "batch d_vocab"]) -> float:
    """
    Given some logits and some answer tokens, return the accuracy according to
    whether the first answer token has higher logit than the second.
    """
    return (answer_logits[:, 0] > answer_logits[:, 1]).float().mean().item()

def patch_training(
    model: HookedTransformer,
    D_train: PatchingDataset,
    D_test: PatchingDataset,
    patcher: Patcher,  # this is trainable
    baseline_patcher: Patcher,  # this is fixed
    n_epochs: int = 30,
    initial_lr: float = 0.1,
    batch_size: int = 20,
    eval_every: int = 5,
):
    """
    General training with a trainable patch: uses D_train to fit the trainable
    patch, and D_test + baseline patcher to evaluate. Returns the metrics:
    """
    optimizer = torch.optim.Adam(patcher.patch_impl.parameters(), lr=initial_lr)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, end_factor=0.1, total_iters=n_epochs
    )
    pbar = tqdm(range(n_epochs))
    criterion = nn.CrossEntropyLoss()
    current_metrics = {}
    full_metrics = defaultdict(list)

    # evaluate the baseline intervention ONCE to avoid recomputing
    with torch.no_grad():
        ld_base_train_baseline, ld_patched_train_baseline, answer_logits_train_baseline = eval_patch_logitdiffs(
            model, D_train, baseline_patcher, batch_size=batch_size
        )
        ld_base_test_baseline, ld_patched_test_baseline, answer_logits_test_baseline = eval_patch_logitdiffs(
            model, D_test, baseline_patcher, batch_size=batch_size
        )
        ld_lost_train_baseline = (ld_base_train_baseline - ld_patched_train_baseline).mean().item()
        ld_lost_test_baseline = (ld_base_test_baseline - ld_patched_test_baseline).mean().item()
        train_alignment_baseline = get_acc_from_logits(
            answer_logits_train_baseline
        )
        test_alignment_baseline = get_acc_from_logits(
            answer_logits_test_baseline
        )
        print(f'baseline train alignment: {train_alignment_baseline:.3f}')
        print(f'baseline test alignment: {test_alignment_baseline:.3f}')

    for epoch in pbar:
        torch.cuda.empty_cache() # for the folks out there with small GPUs :)
        gross_loss = 0
        for batch_dataset in D_train.batches(batch_size=batch_size):
            answer_logits_patched, logits_base, _, logits_patched = patcher.run_patching(
                model, batch_dataset.base.tokens, batch_dataset.source.tokens,
                answer_tokens_base=batch_dataset.base.answer_tokens,
                answer_tokens_source=batch_dataset.source.answer_tokens,
                patched_answer_tokens=batch_dataset.patched_answer_tokens,
                return_full_patched_logits=True,
            )
            # encode as one-hot
            patched_answers_batch_onehot = F.one_hot(
                batch_dataset.patched_answer_tokens[:, 0].long(),
                num_classes=logits_patched.shape[-1],
            ).cuda()
            loss = criterion(logits_patched, patched_answers_batch_onehot.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            gross_loss += loss.item() * len(batch_dataset)
        scheduler.step()

        current_metrics["train_loss"] = gross_loss / len(D_train)
        full_metrics["train_loss"].append(current_metrics["train_loss"])

        if epoch % eval_every == 0:
            with torch.no_grad():
                ld_base_train, ld_patched_train, answer_logits_patched_train = eval_patch_logitdiffs(
                    model, D_train, patcher, batch_size=batch_size
                )
                ld_base_test, ld_patched_test, answer_logits_patched_test = eval_patch_logitdiffs(
                    model, D_test, patcher, batch_size=batch_size
                )

                # evaluate the alignment accuracies on train and test                
                current_metrics['train_alignment'] = get_acc_from_logits(
                    answer_logits=answer_logits_patched_train, 
                )
                full_metrics['train_alignment'].append(current_metrics['train_alignment'])
                current_metrics['test_alignment'] = get_acc_from_logits(
                    answer_logits=answer_logits_patched_test,
                )
                full_metrics['test_alignment'].append(current_metrics['test_alignment'])

                # metrics['ld_base_train'] = ld_base_train.mean().item()
                # metrics['ld_patched_train'] = ld_patched_train.mean().item()
                # metrics['ld_base_test'] = ld_base_test.mean().item()
                # metrics['ld_patched_test'] = ld_patched_test.mean().item()
                current_metrics["ld_fraction_recovered_train"] = (
                    ld_base_train - ld_patched_train
                ).mean().item() / ld_lost_train_baseline
                full_metrics["ld_fraction_recovered_train"].append(
                    current_metrics["ld_fraction_recovered_train"]
                )
                current_metrics["ld_fraction_recovered_test"] = (
                    ld_base_test - ld_patched_test
                ).mean().item() / ld_lost_test_baseline
                full_metrics["ld_fraction_recovered_test"].append(
                    current_metrics["ld_fraction_recovered_test"]
                )

        pbar.set_description(f"epoch {epoch}: {metrics_to_str(current_metrics)}")

    torch.cuda.empty_cache()
    return dict(full_metrics)


################################################################################
### batched utils
################################################################################
@batched(args=['prompts'], n_outputs=1, reducer='cat')
def run_with_cache(
    prompts: Any, nodes: List[Node], batch_size: int, model: HookedTransformer,
) -> List[Tensor]:
    """
    Run the model on the given prompts, and return the activations for the
    given nodes.
    """
    if len(prompts) % batch_size != 0:
        raise ValueError(f"Number of prompts ({len(prompts)}) must be a multiple of batch_size ({batch_size})")
    prompt_dataset = PromptDataset(prompts=prompts, model=model)
    _, cache = model.run_with_cache(prompt_dataset.tokens, names_filter=Node.get_names_filter(nodes))
    model.reset_hooks()
    return [node.get_value(cache) for node in nodes]

@batched(args=['prompts', 'answer_tokens'], n_outputs=1, reducer='cat')
def run_with_hooks(
    prompts: Any, hooks: List[Tuple[str, Callable]], batch_size: int, model: HookedTransformer,
    answer_tokens: Tensor, return_predictions: bool = False,
) -> Tensor:
    prompt_dataset = PromptDataset(prompts=prompts, model=model)
    model.reset_hooks()
    logits = model.run_with_hooks(prompt_dataset.tokens, fwd_hooks=hooks)
    if return_predictions:
        return logits[:, -1, :].argmax(dim=-1)
    else:
        return logits[:, -1, :].gather(1, index=answer_tokens.cuda())