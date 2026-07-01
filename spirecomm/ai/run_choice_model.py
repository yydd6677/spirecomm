import os
from pathlib import Path
import re

from spirecomm.ai.observation import canonicalize_serialized_state
from spirecomm.ai.card_reward_model import (
    STATE_DIM,
    build_state_vector,
    clamp_scale,
    normalize_token,
    stable_bucket,
)
from spirecomm.ai.torch_compat import nn, torch

CHOICE_BUCKETS = 256
CHOICE_KIND_ORDER = ["boss_relic", "event", "campfire"]
CHOICE_KIND_INDEX = {name: index for index, name in enumerate(CHOICE_KIND_ORDER)}
CHOICE_EXTRA_DIM = 8
CHOICE_DIM = CHOICE_BUCKETS + len(CHOICE_KIND_ORDER) + CHOICE_EXTRA_DIM

EVENT_LABEL_ALIASES = {
    ("scrapooze", "open"): "Success",
    ("scrapooze", "leave"): "Fled",
    ("shininglight", "enter"): "Entered Light",
    ("shininglight", "leave"): "Ignored",
    ("goldenwing", "pray"): "Card Removal",
    ("goldenwing", "smash"): "Gained Gold",
    ("goldenwing", "leave"): "Ignored",
    ("liarsgame", "agree"): "AGREE",
    ("liarsgame", "leave"): "Ignored",
    ("wemeetagain", "givepotion"): "Gave Potion",
    ("wemeetagain", "paygold"): "Paid Gold",
    ("wemeetagain", "givecard"): "Gave Card",
    ("wemeetagain", "leave"): "Ignored",
    ("facetrader", "touch"): "Touch",
    ("worldofgoop", "gathergold"): "Gather Gold",
    ("worldofgoop", "leaveit"): "Left Gold",
    ("mushrooms", "fight"): "Fought Mushrooms",
    ("mushrooms", "heal"): "Healed and dodged fight",
    ("goldenshrine", "pray"): "Pray",
    ("goldenshrine", "desecrate"): "Desecrate",
    ("goldenshrine", "leave"): "Ignored",
    ("deadadventurer", "search"): "Searched '1' times",
    ("deadadventurer", "leave"): "Searched '0' times",
    ("falling", "loseskill"): "Removed Skill",
    ("falling", "losepower"): "Removed Power",
    ("falling", "loseattack"): "Removed Attack",
    ("fountainofcleansing", "drink"): "Removed Curses",
    ("fountainofcleansing", "leave"): "Ignored",
    ("knowingskull", "potion"): "POTION",
    ("knowingskull", "90gold"): "GOLD",
    ("knowingskull", "colorlesscard"): "CARD",
    ("knowingskull", "leave"): "Ignored",
    ("designer", "adjust"): "Upgraded Two",
    ("designer", "cleanup"): "Single Remove",
    ("designer", "fullservice"): "Full Service",
    ("designer", "punch"): "Punched",
    ("forgottenaltar", "offeridol"): "Gave Idol",
    ("forgottenaltar", "offerblood"): "Shed Blood",
    ("forgottenaltar", "shedblood"): "Shed Blood",
    ("forgottenaltar", "smash"): "Smashed Altar",
    ("cursedtome", "read"): "Obtained Book",
    ("cursedtome", "takebook"): "Obtained Book",
    ("cursedtome", "stop"): "Stopped",
    ("cursedtome", "leave"): "Ignored",
    ("duplicator", "pray"): "Copied",
    ("duplicator", "leave"): "Ignored",
    ("beggar", "give75gold"): "Gave Gold",
    ("beggar", "leave"): "Ignored",
    ("thelibrary", "sleep"): "Heal",
    ("maskedbandits", "payallgold"): "Paid Fearfully",
    ("maskedbandits", "fight"): "Fought Bandits",
    ("nest", "steal99gold"): "Stole From Cult",
    ("nest", "joinsthecult"): "Joined the Cult",
    ("nloth", "offer"): "Traded Relic",
    ("nloth", "leave"): "Ignored",
    ("thejoust", "betagainst"): "Bet on Owner",
    ("thejoust", "betfor"): "Bet on Murderer",
    ("themoaihead", "healtofull"): "Heal",
    ("themoaihead", "offergoldenidol"): "Gave Idol",
    ("themoaihead", "leave"): "Ignored",
    ("tomboflordredmask", "weartheredmask"): "Wore Mask",
    ("tomboflordredmask", "offerallgold"): "Paid",
    ("tomboflordredmask", "leave"): "Ignored",
    ("mindbloom", "iamwar"): "Fight",
    ("mindbloom", "iamawake"): "Upgrade",
    ("mindbloom", "iamrich"): "Gold",
    ("mindbloom", "iamhealthy"): "Heal",
    ("addict", "pay85gold"): "Obtained Relic",
    ("addict", "offergold"): "Obtained Relic",
    ("addict", "takeshame"): "Stole Relic",
    ("addict", "rob"): "Stole Relic",
    ("addict", "leave"): "Ignored",
    ("ghosts", "accept"): "Became a Ghost",
    ("ghosts", "leave"): "Ignored",
    ("vampires", "accept"): "Became a vampire",
    ("vampires", "offerbloodvial"): "Became a vampire (Vial)",
    ("vampires", "leave"): "Ignored",
    ("thecleric", "heal"): "Healed",
    ("thecleric", "purify"): "Card Removal",
    ("goldenidol", "takeinjury"): "Take Wound",
    ("goldenidol", "takedamage"): "Take Damage",
    ("goldenidol", "losemaxhp"): "Lose Max HP",
    ("goldenidol", "ignore"): "Ignored",
    ("goldenidol", "leave"): "Ignored",
    ("accursedblacksmith", "leave"): "Ignored",
    ("purifier", "pray"): "Purged",
    ("purifier", "leave"): "Ignored",
    ("upgradeshrine", "pray"): "Upgraded",
    ("upgradeshrine", "leave"): "Ignored",
    ("transmorgrifier", "transmogrify"): "Transformed",
    ("transmorgrifier", "leave"): "Ignored",
    ("colosseum", "flee"): "Fled From Nobs",
    ("colosseum", "fightnobs"): "Fought Nobs",
    ("mysterioussphere", "open"): "Fight",
    ("mysterioussphere", "leave"): "Ignored",
    ("secretportal", "enterportal"): "Took Portal",
    ("secretportal", "leave"): "Ignored",
    ("thewomaninblue", "ignored"): "Bought 0 Potions",
    ("thewomaninblue", "leave"): "Bought 0 Potions",
    ("thewomaninblue", "punch"): "Bought 0 Potions",
    ("thewomaninblue", "bought0potions"): "Bought 0 Potions",
}


def object_name(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return str(value.get("name") or value.get("id") or value.get("key") or value.get("label") or value.get("text") or "")
    for attr in ("name", "relic_id", "label", "text"):
        if hasattr(value, attr):
            attr_value = getattr(value, attr)
            if attr_value:
                return str(attr_value)
    if hasattr(value, "name"):
        return str(value.name)
    return str(value)


def _strip_event_label_details(label):
    text = str(label or "").strip()
    text = re.sub(r"\[[^\]]*\]", "", text)
    text = re.sub(r"\([^)]*\)", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or str(label or "").strip()


def canonical_event_model_label(event_id, label):
    raw = str(label or "").strip()
    stripped = _strip_event_label_details(raw)
    event_key = normalize_token(event_id)
    normalized_candidates = [normalize_token(raw), normalize_token(stripped)]
    for normalized in normalized_candidates:
        if event_key == "goldenidol":
            if normalized.startswith("take") and "damage" in normalized:
                return "Take Damage"
            if normalized.startswith("lose") and "maxhp" in normalized:
                return "Lose Max HP"
        if event_key == "wemeetagain" and normalized.startswith("pay") and normalized.endswith("gold"):
            return "Paid Gold"
        if event_key == "falling":
            if normalized.startswith("loseattack"):
                return "Removed Attack"
            if normalized.startswith("loseskill"):
                return "Removed Skill"
            if normalized.startswith("losepower"):
                return "Removed Power"
        if event_key == "mushrooms" and normalized.startswith("heal"):
            return "Healed and dodged fight"
        if event_key == "nest" and normalized.startswith("jointhecult"):
            return "Joined the Cult"
        if event_key == "thewomaninblue" and normalized.startswith("punch"):
            return "Bought 0 Potions"
    for candidate in (raw, stripped):
        alias = EVENT_LABEL_ALIASES.get((event_key, normalize_token(candidate)))
        if alias is not None:
            return alias
    return stripped or raw


def canonical_campfire_model_label(label):
    text = str(label or "").strip()
    if normalize_token(text) == "toke":
        return "PURGE"
    return text


def _shop_model_item_kind_and_id(candidate):
    if not isinstance(candidate, dict):
        return "item", object_name(candidate)
    item_kind = str(candidate.get("model_item_kind") or candidate.get("item_kind") or candidate.get("action") or "item")
    item_id = candidate.get("model_item_id")
    if item_id is None:
        item_id = candidate.get("item_id") or candidate.get("key") or candidate.get("name") or ""
    if item_kind == "potion":
        # SlayTheData shop purchases record potions as generic items, not as a
        # separate shop_potion namespace.
        item_kind = "item"
        item_id = candidate.get("potion_id") or item_id or candidate.get("name") or ""
    if item_kind == "card":
        upgrades = int(candidate.get("upgrades") or 0)
        if upgrades > 0:
            base = str(candidate.get("card_id") or candidate.get("item_id") or candidate.get("name") or item_id or "")
            if "+" not in base:
                item_id = f"{base}+{upgrades}"
            # The training reconstruction classifies upgraded shop cards as
            # generic items because the raw SlayTheData key includes +1.
            item_kind = "item"
    return item_kind, item_id


def option_token(kind, candidate):
    if kind == "boss_relic":
        if isinstance(candidate, dict):
            return normalize_token(candidate.get("relic_id") or candidate.get("id") or candidate.get("key") or candidate.get("name"))
        if hasattr(candidate, "relic_id"):
            return normalize_token(candidate.relic_id or candidate.name)
        return normalize_token(object_name(candidate))
    if kind == "campfire":
        if hasattr(candidate, "name"):
            return normalize_token(canonical_campfire_model_label(candidate.name))
        return normalize_token(canonical_campfire_model_label(object_name(candidate)))
    if kind == "event":
        if isinstance(candidate, dict):
            event_id = candidate.get("event_id") or candidate.get("event_name") or ""
            label = (
                candidate.get("model_label")
                or canonical_event_model_label(
                    event_id,
                    candidate.get("label") or candidate.get("key") or candidate.get("text") or candidate.get("choice") or candidate.get("name") or "",
                )
            )
            return normalize_token(str(event_id) + "_" + str(label))
        event_id = getattr(candidate, "event_id", "") or getattr(candidate, "event_name", "") or ""
        label = getattr(candidate, "model_label", "") or canonical_event_model_label(
            event_id,
            getattr(candidate, "label", "") or getattr(candidate, "text", "") or object_name(candidate),
        )
        return normalize_token(str(event_id) + "_" + str(label))
    if kind == "map":
        if isinstance(candidate, dict):
            return normalize_token("map_" + str(candidate.get("symbol") or candidate.get("name") or candidate.get("key") or ""))
        return normalize_token("map_" + str(getattr(candidate, "symbol", "") or object_name(candidate)))
    if kind == "shop":
        if isinstance(candidate, dict):
            item_kind, item_id = _shop_model_item_kind_and_id(candidate)
            return normalize_token("shop_" + str(item_kind) + "_" + str(item_id))
        return normalize_token("shop_" + object_name(candidate))
    if kind == "potion":
        if isinstance(candidate, dict):
            action = candidate.get("action") or "use"
            potion_id = candidate.get("potion_id") or candidate.get("item_id") or candidate.get("key") or candidate.get("name") or ""
            return normalize_token("potion_" + str(action) + "_" + str(potion_id))
        potion_id = getattr(candidate, "potion_id", "") or object_name(candidate)
        return normalize_token("potion_use_" + str(potion_id))
    return normalize_token(object_name(candidate))


def normalize_choice_candidate(kind, candidate):
    if not isinstance(candidate, dict):
        return candidate
    normalized = dict(candidate)
    if kind == "boss_relic":
        normalized.setdefault("relic_id", normalized.get("id") or normalized.get("item_id") or normalized.get("name"))
    elif kind == "event":
        normalized.setdefault("label", normalized.get("name") or normalized.get("text") or normalized.get("choice"))
        normalized.setdefault("event_id", normalized.get("event_name"))
    elif kind == "map":
        normalized.setdefault("symbol", normalized.get("name") or normalized.get("key"))
        normalized.setdefault("x", normalized.get("choice_index"))
        normalized.setdefault("next_symbols", [])
        normalized.setdefault("child_count", len(normalized.get("next_symbols") or []))
    elif kind == "shop":
        normalized.setdefault("item_id", normalized.get("relic_id") or normalized.get("card_id") or normalized.get("name"))
    elif kind == "potion":
        normalized.setdefault("item_id", normalized.get("potion_id") or normalized.get("name"))
    return normalized


def _map_next_symbols_from_candidate(candidate):
    if isinstance(candidate, dict):
        return [str(symbol) for symbol in (candidate.get("next_symbols") or []) if symbol]
    return [str(getattr(child, "symbol", "") or "") for child in getattr(candidate, "children", []) if getattr(child, "symbol", None)]


def _map_choice_extra(kind, candidate):
    if kind != "map":
        return None

    next_symbols = _map_next_symbols_from_candidate(candidate)
    child_count = len(next_symbols)
    if isinstance(candidate, dict):
        x = int(candidate.get("x", 0) or 0)
        floor = int(candidate.get("floor", 0) or 0)
    else:
        x = int(getattr(candidate, "x", 0) or 0)
        y = int(getattr(candidate, "y", -1) or -1)
        floor = y + 1 if y >= 0 else 0

    elite_like = {"E", "E_GREEN", "T", "BOSS", "ACT4_ELITE", "HEART"}
    return [
        clamp_scale(x, 6.0),
        clamp_scale(floor, 54.0),
        clamp_scale(child_count, 6.0),
        clamp_scale(sum(1 for symbol in next_symbols if symbol == "M"), 6.0),
        clamp_scale(sum(1 for symbol in next_symbols if symbol == "?"), 6.0),
        clamp_scale(sum(1 for symbol in next_symbols if symbol == "$"), 6.0),
        clamp_scale(sum(1 for symbol in next_symbols if symbol == "R"), 6.0),
        clamp_scale(sum(1 for symbol in next_symbols if symbol in elite_like), 6.0),
    ]


def _choice_feature_text(kind, candidate):
    if kind == "event":
        if isinstance(candidate, dict):
            event_id = candidate.get("event_id") or candidate.get("event_name") or ""
            label = candidate.get("model_label") or candidate.get("label") or candidate.get("key") or candidate.get("text") or candidate.get("choice") or candidate.get("name") or ""
            return canonical_event_model_label(event_id, label)
        event_id = getattr(candidate, "event_id", "") or getattr(candidate, "event_name", "") or ""
        label = getattr(candidate, "model_label", "") or getattr(candidate, "label", "") or getattr(candidate, "text", "") or object_name(candidate)
        return canonical_event_model_label(event_id, label)
    if kind == "campfire":
        return canonical_campfire_model_label(object_name(candidate))
    return object_name(candidate)


def choice_vector(kind, candidate, state_like=None):
    candidate = normalize_choice_candidate(kind, candidate)
    vector = [0.0] * CHOICE_BUCKETS
    token = option_token(kind, candidate)
    if token:
        vector[stable_bucket(token, CHOICE_BUCKETS)] = 1.0
    kind_features = [0.0] * len(CHOICE_KIND_ORDER)
    if kind in CHOICE_KIND_INDEX:
        kind_features[CHOICE_KIND_INDEX[kind]] = 1.0
    text = _choice_feature_text(kind, candidate)
    extra = [0.0] * CHOICE_EXTRA_DIM
    extra[0] = clamp_scale(len(text), 80.0)
    extra[1] = 1.0 if "REST" in text.upper() else 0.0
    extra[2] = 1.0 if "SMITH" in text.upper() else 0.0
    extra[3] = 1.0 if "DIG" in text.upper() else 0.0
    extra[4] = 1.0 if "LIFT" in text.upper() else 0.0
    extra[5] = 1.0 if "RECALL" in text.upper() else 0.0
    extra[6] = 1.0 if "SKIP" in text.upper() else 0.0
    extra[7] = 1.0 if token else 0.0
    map_extra = _map_choice_extra(kind, candidate)
    if map_extra is not None:
        extra = map_extra
    return vector + kind_features + extra


class RunChoicePolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.state_encoder = nn.Sequential(
            nn.Linear(STATE_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.choice_head = nn.Sequential(
            nn.Linear(128 + CHOICE_DIM, 192),
            nn.ReLU(),
            nn.Linear(192, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def encode_state(self, state_tensor):
        return self.state_encoder(state_tensor)

    def score_choice_with_hidden(self, state_hidden, choice_tensor):
        if choice_tensor.dim() == 2:
            return self.choice_head(torch.cat([state_hidden, choice_tensor], dim=-1)).squeeze(-1)
        _, choice_count, _ = choice_tensor.shape
        repeated_state = state_hidden.unsqueeze(1).expand(-1, choice_count, -1)
        return self.choice_head(torch.cat([repeated_state, choice_tensor], dim=-1)).squeeze(-1)

    def forward(self, state_tensor, choice_tensor):
        return self.score_choice_with_hidden(self.encode_state(state_tensor), choice_tensor)


def pairwise_batch_to_tensors(batch, device):
    state_rows = []
    pos_rows = []
    neg_rows = []
    weights = []
    for item in batch:
        state = item["state"]
        kind = item["kind"]
        state_rows.append(build_state_vector(state))
        pos_rows.append(choice_vector(kind, item["pos_choice"], state))
        neg_rows.append(choice_vector(kind, item["neg_choice"], state))
        weights.append(float(item.get("weight", 1.0)))
    return {
        "state": torch.tensor(state_rows, dtype=torch.float32, device=device),
        "pos_choice": torch.tensor(pos_rows, dtype=torch.float32, device=device),
        "neg_choice": torch.tensor(neg_rows, dtype=torch.float32, device=device),
        "weight": torch.tensor(weights, dtype=torch.float32, device=device),
    }


def option_scores_from_pairwise_batch(model, batch):
    state_hidden = model.encode_state(batch["state"])
    return (
        model.score_choice_with_hidden(state_hidden, batch["pos_choice"]),
        model.score_choice_with_hidden(state_hidden, batch["neg_choice"]),
    )


def save_run_choice_checkpoint(model, output_path, training_summary=None):
    payload = {
        "state_dict": model.state_dict(),
        "metadata": {
            "state_dim": STATE_DIM,
            "choice_dim": CHOICE_DIM,
            "choice_buckets": CHOICE_BUCKETS,
        },
    }
    if training_summary is not None:
        payload["training_summary"] = training_summary
    torch.save(payload, output_path)


def load_run_choice_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    model = RunChoicePolicyNetwork().to(device)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model, checkpoint.get("training_summary", {}) if isinstance(checkpoint, dict) else {}


class RunChoiceSelector:
    def __init__(self, kind, checkpoint_path=None, env_var=None, default_name=None, device=None):
        repo_root = Path(__file__).resolve().parents[2]
        default_path = repo_root / "models" / (default_name or f"{kind}.pt")
        self.kind = kind
        self.checkpoint_path = Path(checkpoint_path or (os.environ.get(env_var) if env_var else None) or default_path)
        self.device = device or os.environ.get("SPIRECOMM_RUN_CHOICE_DEVICE") or os.environ.get("SPIRECOMM_MODEL_DEVICE", "cpu")
        self.model = None
        self.training_summary = {}
        if self.checkpoint_path.exists():
            self.model, self.training_summary = load_run_choice_checkpoint(str(self.checkpoint_path), device=self.device)

    @property
    def available(self):
        return self.model is not None

    def _prior_delta_scores(self, candidates, raw_scores):
        summary = self.training_summary or {}
        expected_mode = f"{self.kind}_prior_delta"
        if summary.get("score_mode") != expected_mode:
            return raw_scores
        log_prior_by_token = summary.get(f"{self.kind}_option_log_prior_by_token") or {}
        if not log_prior_by_token:
            return raw_scores
        if self.kind == "event":
            env_name = "SPIRECOMM_EVENT_PRIOR_WEIGHT_OVERRIDE"
            summary_key = "event_prior_weight"
            default_prior_weight = summary.get(summary_key, 1.0)
        elif self.kind == "shop":
            env_name = "SPIRECOMM_SHOP_PRIOR_WEIGHT_OVERRIDE"
            summary_key = "shop_prior_weight"
            # The selected runtime shop policy uses the tuned prior weight, not
            # the checkpoint's original training-time default.
            default_prior_weight = 0.8
        else:
            return raw_scores
        try:
            prior_weight = float(os.environ.get(env_name, default_prior_weight))
        except (TypeError, ValueError):
            prior_weight = float(default_prior_weight or 1.0)
        prior_values = [
            prior_weight * float(log_prior_by_token.get(option_token(self.kind, candidate), 0.0))
            for candidate in candidates
        ]
        prior_tensor = torch.tensor(prior_values, dtype=raw_scores.dtype, device=raw_scores.device)
        return raw_scores + prior_tensor

    def choose(self, state_like, candidates, *, return_scores: bool = True):
        if not self.available or not candidates:
            return None
        if isinstance(state_like, dict):
            state_like = canonicalize_serialized_state(state_like)
        normalized_candidates = [normalize_choice_candidate(self.kind, candidate) for candidate in candidates]
        state_tensor = torch.tensor([build_state_vector(state_like)], dtype=torch.float32, device=self.device)
        choice_tensor = torch.tensor(
            [[choice_vector(self.kind, candidate, state_like) for candidate in normalized_candidates]],
            dtype=torch.float32,
            device=self.device,
        )
        with torch.inference_mode():
            scores = self.model(state_tensor, choice_tensor)[0]
            scores = self._prior_delta_scores(normalized_candidates, scores)
        index = int(torch.argmax(scores).item())
        return {
            "choice_index": index,
            "scores": [float(value) for value in scores.detach().cpu().tolist()] if return_scores else [],
        }


class BossRelicSelector(RunChoiceSelector):
    def __init__(self, checkpoint_path=None, device=None):
        super().__init__(
            "boss_relic",
            checkpoint_path=checkpoint_path,
            env_var="SPIRECOMM_BOSS_RELIC_MODEL_PATH",
            default_name="boss_relic.pt",
            device=device,
        )


class EventChoiceSelector(RunChoiceSelector):
    def __init__(self, checkpoint_path=None, device=None):
        super().__init__(
            "event",
            checkpoint_path=checkpoint_path,
            env_var="SPIRECOMM_EVENT_CHOICE_MODEL_PATH",
            default_name="event_choice.pt",
            device=device,
        )


class CampfireChoiceSelector(RunChoiceSelector):
    def __init__(self, checkpoint_path=None, device=None):
        super().__init__(
            "campfire",
            checkpoint_path=checkpoint_path,
            env_var="SPIRECOMM_CAMPFIRE_MODEL_PATH",
            default_name="campfire.pt",
            device=device,
        )


class MapChoiceSelector(RunChoiceSelector):
    def __init__(self, checkpoint_path=None, device=None):
        super().__init__(
            "map",
            checkpoint_path=checkpoint_path,
            env_var="SPIRECOMM_MAP_CHOICE_MODEL_PATH",
            default_name="map_choice.pt",
            device=device,
        )


class ShopChoiceSelector(RunChoiceSelector):
    def __init__(self, checkpoint_path=None, device=None):
        super().__init__(
            "shop",
            checkpoint_path=checkpoint_path,
            env_var="SPIRECOMM_SHOP_CHOICE_MODEL_PATH",
            default_name="shop_choice_prior_delta.pt",
            device=device,
        )


class PotionUseSelector(RunChoiceSelector):
    def __init__(self, checkpoint_path=None, device=None):
        super().__init__(
            "potion",
            checkpoint_path=checkpoint_path,
            env_var="SPIRECOMM_POTION_USE_MODEL_PATH",
            default_name="potion_use.pt",
            device=device,
        )
