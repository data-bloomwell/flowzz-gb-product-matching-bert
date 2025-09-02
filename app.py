import pandas as pd
import re
import os
import torch
from sentence_transformers import SentenceTransformer, util

import logging
from datetime import datetime as dt, timedelta

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend

import boto3
from botocore.exceptions import ClientError

from snowflake.connector.pandas_tools import write_pandas

import snowflake.connector as sc
import sys


def lambda_handler(event, context):

    logger = logging.getLogger(__name__)

    def get_secret(secret_name):
            secret_name = secret_name
            region_name = "eu-central-1"

            session = boto3.session.Session()
            client = session.client(
                    service_name='secretsmanager',
                    region_name=region_name
            )

            try:
                    get_secret_value_response = client.get_secret_value(
                    SecretId=secret_name
                    )
            except ClientError as e:
                    raise e

            secret = get_secret_value_response['SecretString']
            return secret

    private_key_pem = get_secret("sf_user_rsa_key")
    sf_account = get_secret("sf_account")
    secret_file_pass = get_secret("sf_user_rsa_key_pass")


    # Load the private key directly in memory (since it's unencrypted)
    private_key = serialization.load_pem_private_key(
    private_key_pem.encode("utf-8"),
    password=secret_file_pass.encode("utf-8"),  # No passphrase required
    backend=default_backend(),
    )


    # Convert to the required format
    pkb = private_key.private_bytes(
    encoding=serialization.Encoding.DER,
    format=serialization.PrivateFormat.PKCS8,
    encryption_algorithm=serialization.NoEncryption(),
    )


    # ********************Creating the connection to Snowflake.******************************* #

    try:

            cnn = sc.connect(
                    account=sf_account,
                    user="DATA_INGESTION",
                    private_key=pkb,
                    warehouse='COMPUTE_WH',
                    database='EXTERNAL_DATA_STORAGE',
                    schema='FLOWZZ_SCRAPPED'
            )

            cursor = cnn.cursor()
            print("DB Connection is Created.")

    except Exception as e:
            print('Error occurred during connection:', e)
            logger.error("ERROR: %s", str(e))
            sys.exit()

    try:
        # ---------------- Query 1: GRUENE_BRISE products ----------------
        cursor.execute("SELECT * FROM GRUENE_BRISE.A_LZ_SALES.ALL_PRODUCTS;")
        gb = cursor.fetchall()
        columns = [col[0] for col in cursor.description]
        gb = pd.DataFrame(gb, columns=columns)
        print("Latest Gruene Brise Products:", gb.shape)

        # ---------------- Query 2: Unmatched FLOWZZ_GB_MATCHED_PRODUCTS products to match----------------
        query_unmatched = """
        SELECT *
        FROM (
            SELECT *,
                ROW_NUMBER() OVER (
                    PARTITION BY FLOWZ_PRODUCT, FLOWZ_STRAIN
                    ORDER BY DATA_EXTRACTION_TIME DESC
                ) AS rn
            FROM EXTERNAL_DATA_STORAGE.FLOWZZ_SCRAPPED.FLOWZZ_GB_MATCHED_PRODUCTS
            WHERE MATCHING_SCORE IS NULL
            AND GB_PRODUCT_ID IS NULL
        ) sub
        WHERE rn = 1
        ORDER BY DATA_EXTRACTION_TIME DESC;
        """
        cursor.execute(query_unmatched)
        unmatched_products = cursor.fetchall()
        columns = [col[0] for col in cursor.description]
        flowzz = pd.DataFrame(unmatched_products, columns=columns)
        print("Flowzz Products to match:", flowzz.shape)

    except Exception as e:
        print('Error occurred during query:', e)
        logger.error("ERROR: %s", str(e))
        sys.exit()


    # wholesaler / manufacturer columns available for checks, if any missing, create empty columns

    if 'WHOLESALER' not in flowzz.columns:
        flowzz['WHOLESALER'] = ""
    if 'MANUFACTURER_NAME' not in gb.columns:
        gb['MANUFACTURER_NAME'] = ""


    # -----------------------------
    # cleaning 
    # -----------------------------


    country = {'CA': 'Kanada', 'DK': 'Dänemark', 'PT': 'Portugal', 'LS': 'Lesotho', 'CO': 'Kolumbien', 'DE': 'Deutschland', 'UG': 'Uganda', 'ZA': 'Südafrika', 'NL': 'Niederlande', 'ES': 'Spanien', 'UY': 'Uruguay',
        'AU': 'Australien', 'MK': 'Nordmazedonien', 'NZ': 'Neuseeland', 'GB': 'England', 'IL': 'Israel', 'CZ': 'Tschechien', 'JE': 'Jersey', 'ZW': 'Zimbabwe', 'AF': 'Afghanistan', 'GR': 'Griechenland', 'CH': 'Schweiz'}
    gb['ORIGIN'].replace(country, inplace=True)

    flowzz['ORIGIN'] = flowzz['ORIGIN'].replace({'kanda': 'kanada', '-': 'Deutschland'}, inplace=True)

    def is_number_token(token: str) -> bool:
        # 22, 22/1, 22-1, 22:1, thc22, THC21, etc. most thc cases are considered
        return re.match(r'^\d+[/:,-]?\d*$|^thc\d+', str(token).lower()) is not None

    def tokenize_and_filter(text: str):
        tokens = re.findall(r'\b\w+\b', str(text).lower())
        char_tokens = sorted([t for t in tokens if not is_number_token(t)])
        number_tokens = sorted([t for t in tokens if is_number_token(t)])
        return " ".join(char_tokens), " ".join(number_tokens)

    def clean_and_tokenize(product: str, strain: str, source=None):
        combined = f"{product} {strain}".lower()
        if source == "flowzz":
            combined = combined.replace("cannabisblüten", "")
        # make 'thc23' -> 'thc 23'
        combined = re.sub(r'\b(thc|cbd)(\d+)\b', r'\1 \2', combined)
        # keep only the first part of ratios '22/1' -> '22'
        combined = re.sub(r'(\d+)[/:,-](\d+)', r'\1', combined)
        # normalize
        combined = re.sub(r"[^a-z0-9\s]", " ", combined)
        combined = re.sub(r"\s+", " ", combined).strip()
        return tokenize_and_filter(combined)

    def clean_thc_value(value):
        try:
            value = str(value).replace(',', '.').replace('%', '').strip()
            return float(value)
        except:
            return None

    def is_thc_similar(thc1, thc2, tolerance=1.0):
        t1 = clean_thc_value(thc1)
        t2 = clean_thc_value(thc2)
        if t1 is None or t2 is None:
            return None  # unknown / not applicable
        return abs(t1 - t2) <= tolerance

    def norm(s):
        return str(s).strip().lower()

    # -----------------------------
    # Tokenization columns
    # -----------------------------
    gb[['char_tokens', 'num_tokens']] = gb.apply(
        lambda x: pd.Series(clean_and_tokenize(x.get('FULLNAME', ''), x.get('STRAIN_NAME', ''), source="gb")),
        axis=1
    )
    flowzz[['char_tokens', 'num_tokens']] = flowzz.apply(
        lambda x: pd.Series(clean_and_tokenize(x.get('FLOWZ_PRODUCT', ''), x.get('FLOWZ_STRAIN', ''), source="flowzz")),
        axis=1
    )



    # -----------------------------
    # Load BERT model
    # -----------------------------
    os.environ["TRANSFORMERS_NO_TF"] = "1"
    try:
        model = SentenceTransformer('/var/task/minilm_model')
        logger.debug("MiniLM model loaded.")  
    except Exception as e:
        logger.error(f"Model failed to load: {e}")
        raise

    # -----------------------------
    # Matching parameters
    # -----------------------------
    SIM_THRESHOLD = 0.80   # high-confidence semantic match
    THC_TOLERANCE = 1.0    # ±1% allowed

    # When score < SIM_THRESHOLD,
    # we require all applicable constraints to be TRUE:
    # - exact strain match (if both present)
    # - THC similarity (if both present)
    # - wholesaler==manufacturer (if both present)
    # - num_tokens equality (if both present; we already prefilter, but we recheck)

    def all_applicable_criteria_pass(flow_row, gb_row) -> (bool, dict):
        checks = {}

        # strain
        f_strain = flow_row.get('FLOWZ_STRAIN', '')
        g_strain = gb_row.get('STRAIN_NAME', '')
        strain_applicable = bool(str(f_strain).strip()) and bool(str(g_strain).strip())
        if strain_applicable:
            checks['strain_exact'] = (norm(f_strain) == norm(g_strain))

        # THC
        f_thc = flow_row.get('THC_VALUE', '')
        g_thc = gb_row.get('THC_PERCENTAGE_TO', '')
        thc_cmp = is_thc_similar(f_thc, g_thc, tolerance=THC_TOLERANCE)
        # thc_cmp is True/False/None (None means not applicable)
        if thc_cmp is not None:
            checks['thc_close'] = thc_cmp

        # wholesaler(manual) == manufacturer(GB)
        f_wh = flow_row.get('WHOLESALER', '')
        g_mf = gb_row.get('MANUFACTURER_NAME', '')
        wh_applicable = bool(str(f_wh).strip()) and bool(str(g_mf).strip())
        if wh_applicable:
            checks['wh_equals_mf'] = (norm(f_wh) == norm(g_mf))



        f_og = flow_row.get('ORIGIN', '')
        g_og= gb_row.get('ORIGIN', '')
        og_applicable = bool(str(f_og).strip()) and bool(str(g_og).strip())
        if og_applicable:
            checks['og_equals_og'] = (norm(f_og) == norm(g_og))

        # number tokens
        f_num = flow_row.get('num_tokens', '')
        g_num = gb_row.get('num_tokens', '')
        num_applicable = bool(str(f_num).strip()) and bool(str(g_num).strip())
        if num_applicable:
            checks['num_tokens_equal'] = (norm(f_num) == norm(g_num))

        # Require all applicable checks to be True
        all_ok = all(checks.values()) if checks else True
        return all_ok, checks

    # -----------------------------
    # matching loop 
    # -----------------------------
    results = []

    for i in range(len(flowzz)):
        try:
            f_row = flowzz.iloc[i]
            flowz_char = f_row['char_tokens']
            flowz_num  = f_row['num_tokens']

            # Start with strict numeric filtering. If empty, allow all GB.
            if str(flowz_num).strip():
                # allow GB with equal num_tokens
                candidates = gb[(gb['num_tokens'] == flowz_num)]
                if candidates.empty:
                    # fallback: allow GB with no number tokens
                    candidates = gb[(gb['num_tokens'] == "") | (gb['num_tokens'].isna())]
            else:
                candidates = gb.copy()

            flowz_embed = model.encode(flowz_char, convert_to_tensor=True)

            if candidates.empty:
                results.append({
                    'flowz_product': f_row.get('FLOWZ_PRODUCT', ''),
                    'flowz_strain': f_row.get('FLOWZ_STRAIN', ''),
                    'flowz_thc': f_row.get('THC_VALUE', ''),
                    'flowz_wholesaler': f_row.get('WHOLESALER', ''),
                    'flowz_origin': f_row.get('ORIGIN', ''),
                    'gb_product_id': None,
                    'gb_product': None,
                    'gb_strain': None,
                    'gb_manufacturer': None,
                    'gb_thc': None,
                    'matching_score': 0.0,
                    'reason': 'no candidate'
                })
                continue

            # Embed candidates' char_tokens
            cand_embeds = model.encode(candidates['char_tokens'].tolist(), convert_to_tensor=True)
            cos_scores = util.cos_sim(flowz_embed, cand_embeds)[0]
            best_idx = torch.argmax(cos_scores).item()
            best_score = float(cos_scores[best_idx])
            best_match = candidates.iloc[best_idx]

        
            if best_score >= SIM_THRESHOLD:
                reason = "high score match"
                accept = True
                checks = {}
            else:
                # Apply simultaneous constraints
                accept, checks = all_applicable_criteria_pass(f_row, best_match)
                reason = "low score + all-criteria-pass" if accept else "low score + criteria-fail"

            if accept:
                results.append({
                    'flowz_product': f_row.get('FLOWZ_PRODUCT', ''),
                    'flowz_strain': f_row.get('FLOWZ_STRAIN', ''),
                    'flowz_thc': f_row.get('THC_VALUE', ''),
                    'flowz_wholesaler': f_row.get('WHOLESALER', ''),
                    'flowz_origin': f_row.get('ORIGIN', ''),
                    'gb_product_id': best_match.get('ID', None),
                    'gb_product': best_match.get('FULLNAME', ''),
                    'gb_strain': best_match.get('STRAIN_NAME', ''),
                    'gb_manufacturer': best_match.get('MANUFACTURER_NAME', ''),
                    'gb_thc': best_match.get('THC_PERCENTAGE_TO', ''),
                    'gb_origin': best_match.get('ORIGIN', ''),
                    'matching_score': round(best_score, 4),
                    'reason': reason,
                    'criteria': checks
                })
            else:
                results.append({
                    'flowz_product': f_row.get('FLOWZ_PRODUCT', ''),
                    'flowz_strain': f_row.get('FLOWZ_STRAIN', ''),
                    'flowz_thc': f_row.get('THC_VALUE', ''),
                    'flowz_wholesaler': f_row.get('WHOLESALER', ''),
                    'flowz_origin': f_row.get('ORIGIN', ''),
                    'gb_product_id': None,
                    'gb_product': None,
                    'gb_strain': None,
                    'gb_manufacturer': None,
                    'gb_thc': None,
                    'gb_origin': None,
                    'matching_score': 0.0,
                    'reason': reason,
                    'criteria': checks
                })

        except Exception as e:
            logger.warning(f"Row {i} failed for product={f_row.get('FLOWZ_PRODUCT', '')}: {e}")
            continue



    # -----------------------------
    # Finalize results
    # -----------------------------
    flowzz_gb_matched = pd.DataFrame(results)

    print('---------------------------------')


    try:
        if 'matching_score' in flowzz_gb_matched.columns:
            total_products = len(flowzz_gb_matched)
            matched_above_threshold = flowzz_gb_matched[flowzz_gb_matched['matching_score'] >= 0.80].shape[0]
            matched_below_threshold = total_products - matched_above_threshold

            matched_percent = (matched_above_threshold / total_products) * 100 if total_products > 0 else 0

            print(f"[SUMMARY] Total products: {total_products}")
            print(f"[SUMMARY] Matched >=80%: {matched_above_threshold} ({matched_percent:.2f}%)")
            print(f"[SUMMARY] Below 80% (need manual matching): {matched_below_threshold}")

    except Exception as e:
        logger.error(f"Failed to compute summary stats: {e}")


    if 'gb_product_id' in flowzz_gb_matched.columns:
        flowzz_gb_matched['gb_product_id'] = flowzz_gb_matched['gb_product_id'].astype('Int64').astype(str)


    if 'criteria' in flowzz_gb_matched.columns:
        crit_df = flowzz_gb_matched['criteria'].apply(lambda d: {} if not isinstance(d, dict) else d).apply(pd.Series)
        flowzz_gb_matched = pd.concat([flowzz_gb_matched.drop(columns=['criteria']), crit_df], axis=1)

    flowzz_gb_matched.sort_values(by='matching_score', ascending=False, inplace=True)
    flowzz_gb_matched = flowzz_gb_matched.reset_index(drop=True)
    flowzz_gb_matched.columns = flowzz_gb_matched.columns.str.upper()
    flowzz_gb_matched['MATCHED_AT'] = dt.now().isoformat()

    print("Final matched results:", flowzz_gb_matched.shape)

    print('----------------------------------')


    try:
        success, nchunks, nrows, _ = write_pandas(cnn, flowzz_gb_matched, 'FLOWZZ_GB_BERT_MATCHES', chunk_size=15000, overwrite=False)
        print(f"Load Status: {success}, Data Chunks {nchunks}, Data Rows {nrows}")  
    except Exception as e:
        logger.error("ERROR: %s", str(e))
        sys.exit()

    
    cursor.close()
    cnn.close()