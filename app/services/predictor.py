import pandas as pd
import nltk
from sentence_transformers import util
from ..config import settings
from .state import state

def _filter_by_ects(df, current_ects):
    filtered_df = df.copy()
    if current_ects:
        filtered_df['studycredit_num'] = pd.to_numeric(filtered_df['studycredit'], errors='coerce').fillna(0)
        filtered_df = filtered_df[filtered_df['studycredit_num'] == current_ects]
    return filtered_df

def _calculate_semantic_scores(student, filtered_df):
    tags_text = ", ".join(student.tags) if student.tags else ""
    input_text = f"Tags: {tags_text}. Description: {student.description}"
    student_embedding = state.model.encode(input_text, convert_to_tensor=True)

    filtered_indices = filtered_df.index.tolist()
    relevant_embeddings = state.module_embeddings[filtered_indices]
    
    cosine_scores = util.cos_sim(student_embedding, relevant_embeddings)[0]
    return cosine_scores.cpu().numpy(), student_embedding

def _apply_location_boost(scores, filtered_df, preferred_location):
    if preferred_location and preferred_location != "Geen":
        loc_matches = filtered_df['location'].str.contains(preferred_location, case=False, na=False)
        scores[loc_matches] += settings.LOCATION_BOOST
    return scores

def _calculate_single_tag_boost(row, user_tags):
    boost = 0.0
    row_tags = []
    
    def extract(val):
        if isinstance(val, list): return val
        if isinstance(val, str):
            try:
                import ast
                res = ast.literal_eval(val)
                if isinstance(res, list): return res
                return [val]
            except (ValueError, SyntaxError):
                if "," in val: return [x.strip() for x in val.split(",")]
                return [val]
            except Exception:
                return [val]
        return []

    row_tags.extend(extract(row.get('module_tags_en', [])))
    row_tags.extend(extract(row.get('module_tags_nl', [])))
    
    row_tags_lower = [str(t).lower() for t in row_tags]
    
    for ut in user_tags:
        if any(ut in rt for rt in row_tags_lower): 
            boost += settings.TAG_BOOST_PER_MATCH
    
    return min(boost, settings.MAX_TAG_BOOST)

def _apply_tag_boost(scores, filtered_df, student_tags):
    if student_tags:
        user_tags = [t.lower() for t in student_tags]
        tag_boosts = filtered_df.apply(lambda row: _calculate_single_tag_boost(row, user_tags), axis=1).to_numpy()
        scores += tag_boosts
    return scores

def _get_ai_reason(row, student_embedding, language, lang_suffix):
    desc_col = f'description{lang_suffix}'
    short_desc_col = f'shortdescription{lang_suffix}'
    
    description_text = row.get(desc_col, "") or row.get('description_en', "")
    short_description_text = row.get(short_desc_col, "") or row.get('shortdescription_en', "")

    full_text = f"{short_description_text} {description_text}"
    sentences = nltk.sent_tokenize(full_text)
    
    ai_reason = "Match based on general profile overlap." if language.upper() != "NL" else "Match op basis van je profiel."

    if sentences:
        sent_embeddings = state.model.encode(sentences, convert_to_tensor=True)
        sent_scores = util.cos_sim(student_embedding, sent_embeddings)[0]
        best_sent_idx = sent_scores.argmax().item()
        best_sentence = sentences[best_sent_idx].strip()
        
        if len(best_sentence) > 200:
            best_sentence = best_sentence[:197] + "..."
        
        prefix = "AI-Inzicht" if language.upper() == "NL" else "AI-Insight"
        ai_reason = f'💡 {prefix}: "...{best_sentence}..."'
    
    return ai_reason, description_text, short_description_text

def _format_single_result(row, score, student_embedding, language, lang_suffix):
    ai_reason, desc_text, short_desc_text = _get_ai_reason(row, student_embedding, language, lang_suffix)
    
    name_col = f'name{lang_suffix}'
    module_name = row.get(name_col, row['name_en'])

    return {
        "ID": str(row['_id']),
        "Module_Name": module_name,
        "Description": short_desc_text if short_desc_text else desc_text[:100] + "...",
        "Score": round(float(score), 2),
        "AI_Reason": ai_reason,
        "Details": {
            "ects": int(row['studycredit']) if pd.notna(row['studycredit']) else 0,
            "location": str(row['location'])
        }
    }

def predict_recommendations(student, language="NL"):
    # 1. Filter
    filtered_df = _filter_by_ects(state.df, student.current_ects)
    
    if filtered_df.empty:
         return {
            "recommendations": [],
            "number_of_results": 0,
            "status": "success",
            "message": "No modules found with this ECTS."
        }

    # 2. Scores
    scores, student_embedding = _calculate_semantic_scores(student, filtered_df)
    scores = _apply_location_boost(scores, filtered_df, student.preferred_location)
    scores = _apply_tag_boost(scores, filtered_df, student.tags)

    # 3. Sort & Format
    top_k = min(5, len(filtered_df))
    top_indices = scores.argsort()[-top_k:][::-1]
    
    lang_suffix = "_en" if language.upper() == "EN" else "_nl"
    results = []
    
    for idx in top_indices:
        idx = int(idx)
        results.append(_format_single_result(
            filtered_df.iloc[idx], 
            scores[idx], 
            student_embedding, 
            language, 
            lang_suffix
        ))

    return {
        "recommendations": results,
        "number_of_results": len(results),
        "status": "success"
    }
