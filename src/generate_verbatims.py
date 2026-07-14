"""Synthetic customer verbatim generation (challenge brief section 4.4).

Generates one short "last interaction note" per customer (1-3 sentences,
in the voice of the customer), conditioned on tenure/contract/services and
the customer's NPS label. ~17.5% of verbatims are deliberately
"discordant" — a tone that doesn't match the label (a polite Detractor, a
Promoter grumbling about one detail) — per the challenge's requirement
for imperfect text/label correlation.

Provider: the brief (section 4.4) allows "an LLM of your choice." The
final, committed `verbatims.csv` is generated with Google Gemini
(`VERBATIM_PROVIDER=gemini`, the default) via batched calls — see D29 for
why. Groq (`VERBATIM_PROVIDER=groq`) was used earlier to validate the
whole pipeline for free (prompt, discordant-tone mechanism, downstream
embedding/retraining/fairness code); an Anthropic path
(`VERBATIM_PROVIDER=anthropic`) also exists from an earlier plan. Swapping
provider is a one-line `.env` change, never a code change. See
docs/decisions.md for why this isn't a silent provider mix.

Reproducibility note: none of the three providers' APIs has a `seed`
parameter usable the way OpenAI's does, so exact text reproducibility is
not achievable even at temperature=0. What IS deterministic: the prompt
template, the conditioning data pulled per customer, and — via
`config.SEED` — which customers get the discordant treatment and how the
pilot sample is drawn. The committed CSV is the reproducible artifact;
re-running this script selects the same customers for the same tone
treatment but may generate different exact wording.

Phases, run explicitly (never automatically chained):
1. `build_pilot()` — ~50 customers, synchronous single-item calls, for
   human review. Works with any provider.
2. `run_full_generation()` — all customers, concurrent single-item calls,
   resumable, retries on rate limits. Provider-agnostic (Groq/Anthropic);
   this is NOT the path used for Gemini (see 4) — one request per
   customer makes the free tier's ~1,500 requests/day quota impossible to
   clear for ~7043 customers in reasonable time.
3. `run_full_batch_anthropic()` — all customers, via the Anthropic
   Message Batches API (50% cheaper, async). From the earlier
   Anthropic-based plan; unused now that the provider is Gemini.
4. `run_full_generation_gemini()` — all customers, via Gemini, batched
   (config.GEMINI_BATCH_SIZE customers generated per API call, structured
   JSON output). This is the intended path for the final, committed run —
   see D29.
"""
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from dotenv import load_dotenv

import config

# French per the user's explicit instruction (session of 2026-07-12, see
# docs/decisions.md D29) — applies to every provider, not just Gemini,
# since the language is a content decision independent of which vendor
# API produces the text.
SYSTEM_PROMPT = (
    "Vous générez de courtes notes d'interaction client synthétiques et "
    "réalistes pour un jeu de données d'analyse interne d'un opérateur "
    "télécom (un exercice de data science basé sur le jeu de données IBM "
    "Telco Customer Churn). Chaque note simule ce qu'un client a dit ou "
    "écrit lors de son dernier contact avec le support : résumé d'appel "
    "téléphonique, extrait de chat, ou avis laissé sur une application. "
    "Les notes doivent sonner comme un vrai client télécom francophone et "
    "s'ancrer dans les détails de compte fournis, sans rester génériques. "
    "Limitez chaque note à 1-3 phrases, entièrement en français."
)

USER_TEMPLATE = """\
Profil client :
- Ancienneté : {tenure} mois
- Contrat : {contract}
- Charges mensuelles : {monthly_charges:.2f} $
- Services actifs : {services}

{tone_instruction}

Rédigez UNE courte note d'interaction client (1 à 3 phrases, à la première \
personne, comme si vous résumiez un appel, un chat, ou un avis \
d'application). Ne mentionnez jamais les mots "Detractor", "Passive", \
"Promoter", "NPS", ni aucun score ou label interne. N'inventez pas de nom \
ou de numéro de compte. Répondez UNIQUEMENT avec le texte de la note, \
rien d'autre (pas de préambule, pas de guillemets)."""

CONTRACT_FR = {
    "Month-to-month": "sans engagement (mensuel)",
    "One year": "un an",
    "Two year": "deux ans",
}

TONE_INSTRUCTIONS = {
    ("Detractor", False): (
        "Ce client est mécontent (faible satisfaction). Écrivez sur un ton "
        "frustré et agacé — une plainte authentique ancrée dans les "
        "détails du compte ci-dessus. Choisissez UN angle précis plutôt "
        "que d'en lister plusieurs : le prix, un problème de fiabilité "
        "spécifique, un sentiment de ne pas être soutenu par un agent, un "
        "blocage contractuel, ou une meilleure offre chez un concurrent. "
        "Variez l'angle choisi — ne partez pas systématiquement sur la "
        "comparaison avec un concurrent ; n'utilisez cet angle que "
        "lorsqu'il correspond le mieux à ce profil précis."
    ),
    ("Detractor", True): (
        "Ce client est mécontent (faible satisfaction), mais écrivez "
        "cette note sur un ton DISCORDANT : poli, calme et résigné plutôt "
        "que en colère — l'insatisfaction est réelle mais atténuée. Pas "
        "de points d'exclamation, pas de ton élevé."
    ),
    ("Passive", False): (
        "Ce client est neutre (satisfaction tiède). Écrivez sur un ton "
        "neutre et factuel — ni éloge ni plainte, juste une note factuelle "
        "sur le dernier contact."
    ),
    ("Passive", True): (
        "Ce client est neutre (satisfaction tiède), mais écrivez cette "
        "note sur un ton DISCORDANT : penchez nettement d'un côté "
        "(légèrement satisfait ou légèrement agacé, à votre choix) plutôt "
        "que de rester neutre — toujours mesuré, pas extrême."
    ),
    ("Promoter", False): (
        "Ce client est ravi (haute satisfaction). Écrivez sur un ton "
        "enthousiaste et positif — un éloge authentique ancré dans les "
        "détails du compte ci-dessus. Variez l'ouverture et la structure "
        "des phrases — ne commencez pas systématiquement par la même "
        "formule (par exemple, évitez de toujours commencer par quelque "
        "chose comme « Je tenais à dire à quel point je suis ravi(e) ») ; "
        "un vrai avis ou résumé d'appel s'ouvrirait de nombreuses façons "
        "différentes."
    ),
    ("Promoter", True): (
        "Ce client est ravi (haute satisfaction) dans l'ensemble, mais "
        "écrivez cette note sur un ton DISCORDANT : globalement "
        "enthousiaste et positif, mais incluez un petit reproche précis ou "
        "une légère contrariété (par exemple, une bizarrerie de "
        "facturation, un court temps d'attente). L'impression générale "
        "doit rester positive."
    ),
}

SERVICE_COLUMNS = [
    "Online Security", "Online Backup", "Device Protection", "Tech Support",
    "Streaming TV", "Streaming Movies", "Streaming Music",
]


def summarize_services(row: pd.Series) -> str:
    parts = []
    if row["Phone Service"] == "Yes":
        parts.append("Phone")
    if row["Internet Service"] != "No":
        parts.append(f"Internet ({row['Internet Service']})")
    parts += [col for col in SERVICE_COLUMNS if row[col] == "Yes"]
    return ", ".join(parts) if parts else "no active services"


def build_verbatim_plan(df: pd.DataFrame, seed: int = config.SEED) -> pd.DataFrame:
    """Deterministic per-customer tone + discordant-flag assignment.

    The only randomness in this whole module lives here, seeded once.
    """
    rng = np.random.RandomState(seed)
    plan = df[[config.ID_COL, config.TARGET_COL]].copy()
    plan.columns = [config.ID_COL, "nps_category"]
    plan["nps_category"] = plan["nps_category"].astype(str)
    plan["is_discordant"] = rng.random_sample(len(plan)) < config.VERBATIM_DISCORDANT_RATE
    plan["tone_instruction"] = [
        TONE_INSTRUCTIONS[(cat, discordant)]
        for cat, discordant in zip(plan["nps_category"], plan["is_discordant"])
    ]
    return plan


def render_user_message(row: pd.Series, plan_row: pd.Series) -> str:
    return USER_TEMPLATE.format(
        tenure=int(row["Tenure Months"]),
        contract=CONTRACT_FR.get(row["Contract"], row["Contract"]),
        monthly_charges=float(row["Monthly Charges"]),
        services=summarize_services(row),
        tone_instruction=plan_row["tone_instruction"],
    )


def save_prompt_used(sample_row: pd.Series, sample_plan_row: pd.Series) -> None:
    provider = get_provider()
    model = config.GROQ_MODEL if provider == "groq" else config.VERBATIM_MODEL
    text = (
        f"PROVIDER: {provider}   MODEL: {model}\n"
        f"temperature={config.VERBATIM_TEMPERATURE}  max_tokens={config.VERBATIM_MAX_TOKENS}\n\n"
        "SYSTEM PROMPT\n" + "=" * 13 + "\n" + SYSTEM_PROMPT + "\n\n"
        "USER MESSAGE TEMPLATE (rendered example below)\n" + "=" * 47 + "\n"
        + USER_TEMPLATE + "\n\n"
        "TONE INSTRUCTIONS BY (nps_category, is_discordant)\n" + "=" * 51 + "\n"
        + "\n\n".join(f"{k}: {v}" for k, v in TONE_INSTRUCTIONS.items()) + "\n\n"
        "EXAMPLE RENDERED USER MESSAGE\n" + "=" * 29 + "\n"
        + render_user_message(sample_row, sample_plan_row) + "\n"
    )
    config.DATA_VERBATIMS_DIR.mkdir(parents=True, exist_ok=True)
    config.VERBATIMS_PROMPT_PATH.write_text(text, encoding="utf-8")
    print(f"Saved prompt template -> {config.VERBATIMS_PROMPT_PATH}")


# --------------------------------------------------------------------------
# Provider abstraction — the ONLY place that knows which LLM API is in use.
# Everything above/below this section works with plain (system, user) text
# in, text out, so swapping VERBATIM_PROVIDER never touches prompt logic.
# --------------------------------------------------------------------------

def get_provider() -> str:
    load_dotenv()
    return os.environ.get("VERBATIM_PROVIDER", config.VERBATIM_PROVIDER_DEFAULT).lower()


def get_client():
    provider = get_provider()
    if provider == "groq":
        from groq import Groq
        if not os.environ.get("GROQ_API_KEY"):
            raise RuntimeError(
                "VERBATIM_PROVIDER=groq but no GROQ_API_KEY found in .env."
            )
        return Groq(api_key=os.environ["GROQ_API_KEY"])

    if provider == "gemini":
        from google import genai
        if not os.environ.get("GEMINI_API_KEY"):
            raise RuntimeError(
                "VERBATIM_PROVIDER=gemini but no GEMINI_API_KEY found in .env."
            )
        return genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    from anthropic import Anthropic
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        raise RuntimeError(
            "No ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN found in .env. Either "
            "add one (see .env.example), or set VERBATIM_PROVIDER=groq with a "
            "GROQ_API_KEY to validate the pipeline for free first."
        )
    return Anthropic()


def clean_output(text: str) -> str:
    text = (text or "").strip()
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1].strip()
    return text


def generate_text(client, system: str, user: str) -> str:
    """One (system, user) -> generated text call, dispatched by provider.

    Single-item path (used by build_pilot()). The real ~7043-customer
    Gemini run uses generate_batch_gemini() instead — see D29 for why
    (free-tier RPD makes ~7000 individual calls impractical).
    """
    provider = get_provider()
    if provider == "groq":
        response = client.chat.completions.create(
            model=config.GROQ_MODEL,
            temperature=config.VERBATIM_TEMPERATURE,
            max_tokens=config.VERBATIM_MAX_TOKENS,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return clean_output(response.choices[0].message.content)

    if provider == "gemini":
        from google.genai import types
        response = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=config.VERBATIM_TEMPERATURE,
                max_output_tokens=config.VERBATIM_MAX_TOKENS,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        return clean_output(response.text)

    response = client.messages.create(
        model=config.VERBATIM_MODEL,
        max_tokens=config.VERBATIM_MAX_TOKENS,
        temperature=config.VERBATIM_TEMPERATURE,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    text = next((b.text for b in response.content if b.type == "text"), "")
    return clean_output(text)


def generate_with_retry(client, system: str, user: str, max_retries: int = 5) -> str:
    """Exponential backoff on rate limits — the 'works even if the free
    tier is limited' guarantee the Groq validation run needs.
    """
    for attempt in range(max_retries):
        try:
            return generate_text(client, system, user)
        except Exception as exc:  # noqa: BLE001 - provider exception types differ
            is_rate_limit = "429" in str(exc) or "rate" in str(exc).lower()
            if not is_rate_limit or attempt == max_retries - 1:
                raise
            delay = min(2 ** attempt * 2, 60)
            time.sleep(delay)
    raise RuntimeError("unreachable")  # loop always returns or raises


# --------------------------------------------------------------------------
# Phase 1: pilot (always synchronous, always small — for human review)
# --------------------------------------------------------------------------

def build_pilot(labeled: pd.DataFrame, plan: pd.DataFrame, n: int = config.VERBATIM_PILOT_SIZE) -> pd.DataFrame:
    """Synchronous calls for a small, stratified, human-reviewable sample."""
    merged = labeled.merge(plan, on=config.ID_COL)
    # Stratified by NPS category (proportional), sampled deterministically.
    # `GroupBy.sample` (not `.apply()`) — pandas 3.x's `.apply()` silently
    # drops the grouping column from each group under `group_keys=False`.
    sample = (
        merged.groupby("nps_category", observed=True)
        .sample(frac=n / len(merged), random_state=config.SEED)
        .reset_index(drop=True)
    )

    client = get_client()
    print(f"Provider: {get_provider()}")
    save_prompt_used(sample.iloc[0], sample.iloc[0])

    rows = []
    for i, row in sample.iterrows():
        user_message = render_user_message(row, row)
        verbatim = generate_with_retry(client, SYSTEM_PROMPT, user_message)
        rows.append({
            config.ID_COL: row[config.ID_COL],
            "nps_category": row["nps_category"],
            "is_discordant": row["is_discordant"],
            "verbatim": verbatim,
        })
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(sample)} pilot verbatims generated")

    pilot_df = pd.DataFrame(rows)
    config.DATA_VERBATIMS_DIR.mkdir(parents=True, exist_ok=True)
    pilot_df.to_csv(config.VERBATIMS_PILOT_CSV_PATH, index=False)
    print(f"Saved -> {config.VERBATIMS_PILOT_CSV_PATH}")
    return pilot_df


# --------------------------------------------------------------------------
# Phase 2: full run, provider-agnostic, concurrent + resumable
# --------------------------------------------------------------------------

def run_full_generation(
    labeled: pd.DataFrame,
    plan: pd.DataFrame,
    max_workers: int = 8,
    checkpoint_every: int = 100,
) -> pd.DataFrame:
    """All customers, via plain concurrent calls to whichever provider is
    configured. Resumable: customers already present in
    `config.VERBATIMS_CSV_PATH` are skipped, so a rate-limited or
    interrupted run can just be re-invoked.
    """
    merged = labeled.merge(plan, on=config.ID_COL)

    done_rows: list[dict] = []
    if config.VERBATIMS_CSV_PATH.exists():
        existing = pd.read_csv(config.VERBATIMS_CSV_PATH)
        done_ids = set(existing[config.ID_COL])
        merged = merged[~merged[config.ID_COL].isin(done_ids)]
        done_rows = existing.to_dict("records")
        print(f"Resuming: {len(done_ids)} already generated, "
              f"{len(merged)} remaining.")

    if len(merged) == 0:
        print("Nothing left to generate.")
        return pd.DataFrame(done_rows)

    client = get_client()
    provider = get_provider()
    print(f"Provider: {provider}  |  generating {len(merged)} verbatims "
          f"with {max_workers} concurrent workers...")

    def _task(row):
        user_message = render_user_message(row, row)
        verbatim = generate_with_retry(client, SYSTEM_PROMPT, user_message)
        return {
            config.ID_COL: row[config.ID_COL],
            "nps_category": row["nps_category"],
            "is_discordant": row["is_discordant"],
            "verbatim": verbatim,
        }

    rows = list(done_rows)
    n_done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_task, row): row[config.ID_COL] for _, row in merged.iterrows()}
        for future in as_completed(futures):
            customer_id = futures[future]
            try:
                rows.append(future.result())
            except Exception as exc:  # noqa: BLE001
                print(f"  [{customer_id}] FAILED after retries: {exc}")
            n_done += 1
            if n_done % checkpoint_every == 0:
                pd.DataFrame(rows).to_csv(config.VERBATIMS_CSV_PATH, index=False)
                print(f"  checkpoint: {n_done}/{len(merged)} this run "
                      f"({len(rows)} total saved)")

    full_df = pd.DataFrame(rows)
    config.DATA_VERBATIMS_DIR.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(config.VERBATIMS_CSV_PATH, index=False)
    print(f"Saved {len(full_df)} verbatims -> {config.VERBATIMS_CSV_PATH} "
          f"(provider={provider})")
    return full_df


# --------------------------------------------------------------------------
# Phase 2b: full run, Gemini-only, batched (config.GEMINI_BATCH_SIZE
# customers per call). The intended path for the final, committed run —
# see D29. One request per customer (like run_full_generation()) would
# need ~7043 calls; Gemini's free tier caps at ~1,500 requests/day, so
# batching is not an optimization here, it's what makes the run feasible
# in one sitting ("sans interruptions", the user's explicit ask).
# --------------------------------------------------------------------------

BATCH_INSTRUCTIONS_HEADER = (
    "Voici {n} profils clients. Pour CHACUN, rédigez UNE note d'interaction "
    "client courte (1 à 3 phrases, en français, à la première personne, "
    "comme si vous résumiez un appel, un chat, ou un avis d'application). "
    "Ne mentionnez jamais les mots \"Detractor\", \"Passive\", \"Promoter\", "
    "\"NPS\", ni aucun score ou label interne. N'inventez pas de nom ou de "
    "numéro de compte. Répondez uniquement avec le JSON structuré demandé : "
    "un élément par client, dans le même ordre, en reprenant exactement "
    "l'identifiant \"id\" donné pour chaque client ci-dessous."
)

CUSTOMER_BLOCK_TEMPLATE = """\
--- Client {id} ---
Ancienneté : {tenure} mois
Contrat : {contract}
Charges mensuelles : {monthly_charges:.2f} $
Services actifs : {services}
Consigne de ton : {tone_instruction}"""


def _gemini_schema():
    """Built lazily (not at import time) so `pydantic` stays an
    implicit dependency of the gemini path only, like `google.genai`."""
    from pydantic import BaseModel

    class VerbatimItem(BaseModel):
        id: str
        text: str

    class VerbatimBatch(BaseModel):
        items: list[VerbatimItem]

    return VerbatimBatch


def render_batch_user_message(batch_df: pd.DataFrame) -> str:
    blocks = [
        CUSTOMER_BLOCK_TEMPLATE.format(
            id=row[config.ID_COL],
            tenure=int(row["Tenure Months"]),
            contract=CONTRACT_FR.get(row["Contract"], row["Contract"]),
            monthly_charges=float(row["Monthly Charges"]),
            services=summarize_services(row),
            tone_instruction=row["tone_instruction"],
        )
        for _, row in batch_df.iterrows()
    ]
    header = BATCH_INSTRUCTIONS_HEADER.format(n=len(batch_df))
    return header + "\n\n" + "\n\n".join(blocks)


def generate_batch_gemini(client, batch_df: pd.DataFrame) -> dict[str, str]:
    """One API call -> {customer_id: verbatim} for every customer in
    `batch_df` the model actually returned an item for. Thinking is
    disabled: this is short creative text generation, not a reasoning
    task, and thinking tokens were observed (empirically, this session)
    consuming most of the output budget and truncating the JSON — see
    D29. max_output_tokens is generous (batch_size x ~300) since the
    65,536 ceiling leaves large headroom and a truncated batch fails the
    whole call, not just one customer.
    """
    from google.genai import types

    VerbatimBatch = _gemini_schema()
    message = render_batch_user_message(batch_df)
    response = client.models.generate_content(
        model=config.GEMINI_MODEL,
        contents=message,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=config.VERBATIM_TEMPERATURE,
            max_output_tokens=max(4000, len(batch_df) * 300),
            response_mime_type="application/json",
            response_schema=VerbatimBatch,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    parsed = VerbatimBatch.model_validate_json(response.text)
    return {item.id: clean_output(item.text) for item in parsed.items if item.text.strip()}


def generate_batch_with_retry(client, batch_df: pd.DataFrame, max_retries: int = 5) -> dict[str, str]:
    for attempt in range(max_retries):
        try:
            return generate_batch_gemini(client, batch_df)
        except Exception as exc:  # noqa: BLE001 - SDK exception types vary
            is_rate_limit = any(s in str(exc) for s in ("429", "RESOURCE_EXHAUSTED")) \
                or "rate" in str(exc).lower()
            if not is_rate_limit or attempt == max_retries - 1:
                raise
            delay = min(2 ** attempt * 5, 90)
            time.sleep(delay)
    raise RuntimeError("unreachable")  # loop always returns or raises


def run_full_generation_gemini(
    labeled: pd.DataFrame,
    plan: pd.DataFrame,
    batch_size: int = config.GEMINI_BATCH_SIZE,
    checkpoint_every_batches: int = 5,
) -> pd.DataFrame:
    """All customers, via Gemini, batch_size customers per call. Resumable
    like run_full_generation(): customers already in
    config.VERBATIMS_CSV_PATH are skipped. A customer missing from a
    batch's response (partial/malformed output) is simply not written —
    it stays eligible for the next invocation, same as a failed batch.
    """
    if get_provider() != "gemini":
        raise RuntimeError(
            "run_full_generation_gemini() requires VERBATIM_PROVIDER=gemini."
        )
    merged = labeled.merge(plan, on=config.ID_COL)

    done_rows: list[dict] = []
    if config.VERBATIMS_CSV_PATH.exists():
        existing = pd.read_csv(config.VERBATIMS_CSV_PATH)
        done_ids = set(existing[config.ID_COL])
        merged = merged[~merged[config.ID_COL].isin(done_ids)]
        done_rows = existing.to_dict("records")
        print(f"Resuming: {len(done_ids)} already generated, "
              f"{len(merged)} remaining.")

    if len(merged) == 0:
        print("Nothing left to generate.")
        return pd.DataFrame(done_rows)

    merged = merged.reset_index(drop=True)
    client = get_client()
    n_batches_total = -(-len(merged) // batch_size)  # ceil division
    print(f"Provider: gemini  |  generating {len(merged)} verbatims in "
          f"{n_batches_total} batches of up to {batch_size} "
          f"(model={config.GEMINI_MODEL})...")

    rows = list(done_rows)
    for batch_num, start in enumerate(range(0, len(merged), batch_size), start=1):
        batch_df = merged.iloc[start:start + batch_size]
        try:
            results = generate_batch_with_retry(client, batch_df)
        except Exception as exc:  # noqa: BLE001
            ids = batch_df[config.ID_COL].tolist()
            print(f"  BATCH {batch_num}/{n_batches_total} FAILED after "
                  f"retries ({len(ids)} customers, will retry next run): {exc}")
            results = {}

        n_missing = 0
        for _, row in batch_df.iterrows():
            cid = row[config.ID_COL]
            if cid in results:
                rows.append({
                    config.ID_COL: cid,
                    "nps_category": row["nps_category"],
                    "is_discordant": row["is_discordant"],
                    "verbatim": results[cid],
                })
            else:
                n_missing += 1
        status = f"  batch {batch_num}/{n_batches_total} done " \
                 f"({len(results)}/{len(batch_df)} customers"
        status += f", {n_missing} missing" if n_missing else ""
        status += f") — {len(rows)} total saved"
        print(status)

        if batch_num % checkpoint_every_batches == 0:
            pd.DataFrame(rows).to_csv(config.VERBATIMS_CSV_PATH, index=False)
            print(f"  checkpoint saved ({len(rows)} rows)")

        if start + batch_size < len(merged):
            time.sleep(config.GEMINI_REQUEST_DELAY_SECONDS)

    full_df = pd.DataFrame(rows)
    config.DATA_VERBATIMS_DIR.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(config.VERBATIMS_CSV_PATH, index=False)
    print(f"Saved {len(full_df)} verbatims -> {config.VERBATIMS_CSV_PATH} "
          f"(provider=gemini, batched)")
    return full_df


# --------------------------------------------------------------------------
# Phase 3: full run, Anthropic-only, via the Message Batches API (50%
# cheaper — the intended path once actually paying for the real run)
# --------------------------------------------------------------------------

def run_full_batch_anthropic(labeled: pd.DataFrame, plan: pd.DataFrame) -> None:
    if get_provider() != "anthropic":
        raise RuntimeError(
            "run_full_batch_anthropic() requires VERBATIM_PROVIDER=anthropic "
            "(the Message Batches API is Anthropic-specific)."
        )
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    client = get_client()
    merged = labeled.merge(plan, on=config.ID_COL)

    requests = [
        Request(
            custom_id=row[config.ID_COL],
            params=MessageCreateParamsNonStreaming(
                model=config.VERBATIM_MODEL,
                max_tokens=config.VERBATIM_MAX_TOKENS,
                temperature=config.VERBATIM_TEMPERATURE,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": render_user_message(row, row)}],
            ),
        )
        for _, row in merged.iterrows()
    ]
    print(f"Submitting a batch of {len(requests)} requests...")
    batch = client.messages.batches.create(requests=requests)
    print(f"Batch ID: {batch.id}  status: {batch.processing_status}")

    while True:
        batch = client.messages.batches.retrieve(batch.id)
        if batch.processing_status == "ended":
            break
        print(f"  status={batch.processing_status}  "
              f"counts={batch.request_counts}")
        time.sleep(30)

    print(f"Batch complete. succeeded={batch.request_counts.succeeded} "
          f"errored={batch.request_counts.errored}")

    plan_lookup = plan.set_index(config.ID_COL)
    rows = []
    for result in client.messages.batches.results(batch.id):
        customer_id = result.custom_id
        if result.result.type == "succeeded":
            msg = result.result.message
            text = next((b.text for b in msg.content if b.type == "text"), "")
            rows.append({
                config.ID_COL: customer_id,
                "nps_category": plan_lookup.loc[customer_id, "nps_category"],
                "is_discordant": plan_lookup.loc[customer_id, "is_discordant"],
                "verbatim": clean_output(text),
            })
        else:
            print(f"  [{customer_id}] {result.result.type}")

    full_df = pd.DataFrame(rows)
    config.DATA_VERBATIMS_DIR.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(config.VERBATIMS_CSV_PATH, index=False)
    print(f"Saved {len(full_df)} verbatims -> {config.VERBATIMS_CSV_PATH} "
          f"(provider=anthropic, batch API)")


if __name__ == "__main__":
    labeled_df = pd.read_parquet(config.LABELED_DATASET_PATH)
    verbatim_plan = build_verbatim_plan(labeled_df)
    print("Verbatim plan distribution:")
    print(verbatim_plan.groupby(["nps_category", "is_discordant"], observed=True).size())

    pilot = build_pilot(labeled_df, verbatim_plan)
    print(f"\nPilot done: {len(pilot)} verbatims. Review before running "
          f"run_full_generation_gemini() (the default provider, batched) "
          f"for the remaining ~{len(labeled_df) - len(pilot)} customers. "
          f"run_full_generation() (Groq/Anthropic, one call per customer) "
          f"and run_full_batch_anthropic() (Anthropic Batches API) remain "
          f"available if VERBATIM_PROVIDER is switched away from gemini.")
