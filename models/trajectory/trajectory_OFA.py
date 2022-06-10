import torch
import numpy as np
import torch.nn.functional as F

from fairseq.distributed import fsdp_wrap

from models.ofa.unify_transformer import base_architecture
from models.ofa.ofa import OFAModel, ofa_base_architecture
from data import data_utils

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024
DEFAULT_MIN_PARAMS_TO_WRAP = int(1e8)


class TrajectoryOFAModel(OFAModel):
    def __init__(self,  args, task):
        self.task = task
        self.args = args

        self.code_image_size = 128

        # make sure all arguments are present in older models
        base_architecture(args)
        ofa_base_architecture(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

        self.pad_idx = self.src_dict.pad()
        self.bos = self.src_dict.bos()
        self.eos = self.src_dict.eos()
        self.bos_item = torch.LongTensor([self.bos])
        self.eos_item = torch.LongTensor([self.eos])

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens = self.build_embedding(args, src_dict, args.encoder_embed_dim, args.encoder_embed_path)
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = self.build_embedding(args, src_dict, args.encoder_embed_dim, args.encoder_embed_path)
            decoder_embed_tokens = self.build_embedding(args, tgt_dict, args.decoder_embed_dim, args.decoder_embed_path)
        #if getattr(args, "freeze_encoder_embedding", False):
        #    encoder_embed_tokens.weight.requires_grad = False
        #if getattr(args, "freeze_decoder_embedding", False):
        #    decoder_embed_tokens.weight.requires_grad = False
        if getattr(args, "offload_activations", False):
            args.checkpoint_activations = True  # offloading implies checkpointing
        encoder = self.build_encoder(args, src_dict, encoder_embed_tokens)
        decoder = self.build_decoder(args, tgt_dict, decoder_embed_tokens)
        if not args.share_all_embeddings:
            min_params_to_wrap = getattr(
                args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP
            )
            # fsdp_wrap is a no-op when --ddp-backend != fully_sharded
            encoder = fsdp_wrap(encoder, min_num_params=min_params_to_wrap)
            decoder = fsdp_wrap(decoder, min_num_params=min_params_to_wrap)

        super().__init__(args, encoder, decoder)
        return

    def decode_value_from_token_index(self, token_index):
        try:
            token = self.tgt_dict[token_index].split('_')[1][:-1]
            v_rel = float(token)/self.args.num_bins
        except:
            return 0.0
        return v_rel

    def decode_value_from_index_tensor(self, index_tensor):
        index_tensor = np.array(index_tensor)

        preds = []
        for i in range(index_tensor.shape[0]):
            pred = []
            for j in range(index_tensor.shape[1] - 1):
                token_index = index_tensor[i][j]
                pred.append(self.decode_value_from_token_index(token_index))
            preds.append(pred)
        return torch.tensor(preds)


    def forward(self, states, actions, rewards, returns_to_go, timesteps, attention_mask=None):

        batch = self.process_traj(states, actions, rewards, returns_to_go, timesteps, attention_mask)

        x, extra = super().forward(batch['net_input']['src_tokens'], batch['net_input']['src_lengths'], batch['net_input']['prev_output_tokens'])

        log_probs = True
        conf = batch['conf'][:, None, None] if 'conf' in batch and batch['conf'] is not None else 1

        if torch.is_tensor(x):
            logits = x.float()
            if log_probs:
                lprobs = F.log_softmax(logits, dim=-1) * conf
            else:
                lprobs = F.softmax(logits, dim=-1) * conf

        # max
        if self.args.action_pred_way == 'max':
            lprobs_max, index_lprobs_max = torch.max(lprobs, dim=2)
            actions_pred = self.decode_value_from_index_tensor(index_lprobs_max)
            #actions_target = self.decode_value_from_index_tensor(batch['target'])
            #print(actions_pred)
            ##print(actions_pred.shape)
            #print('*'*10)
            #print(actions_target)
            #print(actions_target.shape)
            #exit(2)
        # weighted average
        elif self.args.action_pred_way == 'wa':
            raise NotImplementedError
        else:
            raise NotImplementedError

        # get predictions
        actions_pred = actions_pred.reshape((actions.shape[0], actions.shape[1], -1))
        #return_preds = self.predict_return(x[:, 2])  # predict next return given state and action
        #state_preds = self.predict_state(x[:, 2])  # predict next state given state and action
        #action_preds = self.predict_action(x[:, 1])  # predict next action given state

        return None, actions_pred, None

    def process_pure_trajectory(self, states, actions, returns_to_go, index):
        patch_image = torch.zeros((3, self.code_image_size*2, self.code_image_size*2))
        patch_mask = torch.tensor([False])
        code_mask = torch.tensor([False])
        conf = torch.tensor([1.0])

        r_s = torch.cat([returns_to_go[index], states[index]], dim=1)
        a = actions[index]

        uniq_id = str(index)
        state_action_tokens = self.quantize(r_s.reshape(-1), self.args.num_bins)
        action_tokens = self.quantize(a.reshape(-1), self.args.num_bins)

        src_item = torch.cat([self.bos_item, state_action_tokens, self.eos_item])
        target_item = torch.cat([action_tokens, self.eos_item])
        prev_output_item = torch.cat([self.bos_item, action_tokens])

        example = {
            "id": uniq_id,
            "source": src_item,
            "patch_image": patch_image,
            "patch_mask": patch_mask,
            "code_mask": code_mask,
            "target": target_item,
            "prev_output_tokens": prev_output_item,
            "conf": conf,
        }
        return example

    def process_traj(self, states, actions, rewards, returns_to_go, timesteps, attention_mask=None):
        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        rtg_min = torch.min(returns_to_go)
        rtg_max = torch.max(returns_to_go)
        returns_to_go = (returns_to_go - rtg_min)/rtg_max

        examples = []
        for index in range(self.args.batch_size):
            example = self.process_pure_trajectory(states, actions, returns_to_go, index)
            examples.append(example)

        if len(examples) == 0:
            return {}

        def merge(key):
            return data_utils.collate_tokens(
                [s[key] for s in examples],
                self.pad_idx,
                eos_idx=self.eos,
            )

        id = np.array([s["id"] for s in examples])
        src_tokens = merge("source")
        src_lengths = torch.LongTensor([s["source"].ne(self.pad_idx).long().sum() for s in examples])

        patch_images = torch.stack([sample['patch_image'] for sample in examples], dim=0)
        patch_masks = torch.cat([sample['patch_mask'] for sample in examples])

        code_masks = None
        if examples[0].get("code_mask", None) is not None:
            code_masks = torch.cat([sample['code_mask'] for sample in examples])

        conf = torch.cat([s['conf'] for s in examples], dim=0)

        prev_output_tokens = None
        target = None
        if examples[0].get("target", None) is not None:
            target = merge("target")
            tgt_lengths = torch.LongTensor([s["target"].ne(self.pad_idx).long().sum() for s in examples])
            ntokens = tgt_lengths.sum().item()

            if examples[0].get("prev_output_tokens", None) is not None:
                prev_output_tokens = merge("prev_output_tokens")
        else:
            ntokens = src_lengths.sum().item()

        batch = {
            "id": id,
            "nsentences": states.shape[0],
            "ntokens": ntokens,
            "net_input": {
                "src_tokens": src_tokens,
                "src_lengths": src_lengths,
                "patch_images": patch_images,
                "patch_masks": patch_masks,
                "code_masks": code_masks,
                "prev_output_tokens": prev_output_tokens
            },
            "target": target,
            "conf": conf
        }

        '''
        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings        
        '''


        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        #stacked_inputs = torch.stack((returns_embeddings, state_embeddings, action_embeddings), dim=1).permute(0, 2, 1, 3).reshape(batch_size, 3 * seq_length, self.hidden_size)

        return batch

    def quantize(self, tensor_v_rel, num_bins):
        q_tokens = ["<bin_{}>".format(int((v_rel * (num_bins - 1)).round())) for v_rel in tensor_v_rel]
        q_item = self.encode_text(' '.join(q_tokens), use_bpe=False)
        return q_item

    def get_action(self, states, actions, rewards, returns_to_go, timesteps, **kwargs):
        # we don't care about the past rewards in this model

        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.action_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            states = states[:,-self.max_length:]
            actions = actions[:,-self.max_length:]
            returns_to_go = returns_to_go[:,-self.max_length:]
            timesteps = timesteps[:,-self.max_length:]

            # pad all tokens to sequence length
            attention_mask = torch.cat([torch.zeros(self.max_length-states.shape[1]), torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
            states = torch.cat([torch.zeros((states.shape[0], self.max_length-states.shape[1], self.state_dim), device=states.device), states], dim=1).to(dtype=torch.float32)
            actions = torch.cat([torch.zeros((actions.shape[0], self.max_length - actions.shape[1], self.action_dim), device=actions.device), actions], dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat([torch.zeros((returns_to_go.shape[0], self.max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go], dim=1).to(dtype=torch.float32)
            timesteps = torch.cat([torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device), timesteps], dim=1).to(dtype=torch.long)
        else:
            attention_mask = None

        _, action_preds, return_preds = self.forward(states, actions, None, returns_to_go, timesteps, attention_mask=attention_mask, **kwargs)

        return action_preds[0,-1]

    def encode_text(self, text, length=None, append_bos=False, append_eos=False, use_bpe=True):
        s = self.tgt_dict.encode_line(
            line=self.bpe.encode(text) if use_bpe else text,
            add_if_not_exist=False,
            append_eos=False
        ).long()
        if length is not None:
            s = s[:length]
        if append_bos:
            s = torch.cat([self.bos_item, s])
        if append_eos:
            s = torch.cat([s, self.eos_item])
        return s