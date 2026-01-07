import pandas as pd
import nltk
from sentence_transformers import util
from ..config import settings
from .state import state

def predict_recommendations(student, language="NL"):
    df = state.df
    model = state.model
    module_embeddings = state.module_embeddings

    # 1. Hard Filter: ECTS
    filtered_df = df.copy()
    if student.current_ects:
        filtered_df['studycredit_num'] = pd.to_numeric(filtered_df['studycredit'], errors='coerce').fillna(0)
        filtered_df = filtered_df[filtered_df['studycredit_num'] == student.current_ects]
        
        if filtered_df.empty:
             return {
                "recommendations": [],
                "number_of_results": 0,
                "status": "success",
                "message": "No modules found with this ECTS."
            }

    # 2. Embedding
    tags_text = ", ".join(student.tags) if student.tags else ""
    input_text = f"Tags: {tags_text}. Description: {student.description}"
    student_embedding = model.encode(input_text, convert_to_tensor=True)

    # 3. Semantic Matching
    # Select subset of embeddings corresponding to filtered rows
    filtered_indices = filtered_df.index.tolist()
    relevant_embeddings = module_embeddings[filtered_indices]
    
    cosine_scores = util.cos_sim(student_embedding, relevant_embeddings)[0]
    scores = cosine_scores.cpu().numpy()

    # 4. Location Bonus
    if student.preferred_location and student.preferred_location != "Geen":
        loc_matches = filtered_df['location'].str.contains(student.preferred_location, case=False, na=False)
        scores[loc_matches] += settings.LOCATION_BOOST

    # 5. Explicit Tag Boost
    if student.tags:
        user_tags = [t.lower() for t in student.tags]
        
        def calculate_tag_boost(row):
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
                    except:
                        if "," in val: return [x.strip() for x in val.split(",")]
                        return [val]
                return []

            row_tags.extend(extract(row.get('module_tags_en', [])))
            row_tags.extend(extract(row.get('module_tags_nl', [])))
            
            row_tags_lower = [str(t).lower() for t in row_tags]
            
            for ut in user_tags:
                if any(ut in rt for rt in row_tags_lower): 
                    boost += settings.TAG_BOOST_PER_MATCH
            
            return min(boost, settings.MAX_TAG_BOOST)

        tag_boosts = filtered_df.apply(calculate_tag_boost, axis=1).to_numpy()
        scores += tag_boosts

    # 6. Retrieve Top Results
    top_k = min(5, len(filtered_df))
    top_indices_local = scores.argsort()[-top_k:][::-1]

    results = []
    lang_suffix = "_en" if language.upper() == "EN" else "_nl"
    
    for local_idx in top_indices_local:
        local_idx = int(local_idx)
        row = filtered_df.iloc[local_idx]
        score = float(scores[local_idx])
        
        # Explainable AI Logic
        desc_col = f'description{lang_suffix}'
        short_desc_col = f'shortdescription{lang_suffix}'
        
        description_text = row.get(desc_col, "")
        if not description_text: description_text = row.get('description_en', "")
        
        short_description_text = row.get(short_desc_col, "")
        if not short_description_text: short_description_text = row.get('shortdescription_en', "")

        full_text_for_reason = f"{short_description_text} {description_text}"
        sentences = nltk.sent_tokenize(full_text_for_reason)
        
        ai_reason = "Match based on general profile overlap."
        if language.upper() == "NL": ai_reason = "Match op basis van je profiel."

        if sentences:
            sent_embeddings = model.encode(sentences, convert_to_tensor=True)
            sent_scores = util.cos_sim(student_embedding, sent_embeddings)[0]
            best_sent_idx = sent_scores.argmax().item()
            best_sentence = sentences[best_sent_idx].strip()
            
            if len(best_sentence) > 200:
                best_sentence = best_sentence[:197] + "..."
            
            if language.upper() == "NL":
                ai_reason = f'💡 AI-Inzicht: "...{best_sentence}..."'
            else:
                ai_reason = f'💡 AI-Insight: "...{best_sentence}..."'

        loc_str = str(row['location'])
        name_col = f'name{lang_suffix}'
        module_name = row.get(name_col, row['name_en'])

        results.append({
            "ID": str(row['_id']),
            "Module_Name": module_name,
            "Description": short_description_text if short_description_text else description_text[:100] + "...",
            "Score": round(score, 2),
            "AI_Reason": ai_reason,
            "Details": {
                "ects": int(row['studycredit']) if pd.notna(row['studycredit']) else 0,
                "location": loc_str
            }
        })

    return {
        "recommendations": results,
        "number_of_results": len(results),
        "status": "success"
    }
