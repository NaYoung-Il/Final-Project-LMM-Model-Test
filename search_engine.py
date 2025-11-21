import pandas as pd
import numpy as np
import json
import torch
import faiss
import time
import os
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel, CLIPTokenizer, CLIPTextModel

# ==========================================
# [핵심] 평가할 모델 리스트 설정 (확장 가능)
# type: 'sbert' (패키지 사용), 'bert' (Mean Pooling), 'clip' (CLS Token)
# ==========================================
MODEL_CONFIGS = [
    # --- S-BERT 계열 (한국어 문장 유사도 특화) ---
    {"name": "S-BERT (Multitask)", "id": "jhgan/ko-sbert-multitask", "type": "sbert"},
    {"name": "snunlp/KR-SBERT", "id": "snunlp/KR-SBERT-V40K-klueNLI-augSTS", "type": "sbert"},
    {"name": "Multi-MiniLM (Light)", "id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "type": "sbert"},
    
    # --- BERT/RoBERTa 계열 (Mean Pooling 사용) ---
    {"name": "KLUE-BERT Base", "id": "klue/bert-base", "type": "bert"},
    {"name": "KLUE-RoBERTa Base", "id": "klue/roberta-base", "type": "bert"},
    {"name": "KcBERT (Comments)", "id": "beomi/kcbert-base", "type": "bert"},
    {"name": "Google-BERT (Uncased)", "id": "google-bert/bert-base-uncased", "type": "bert"},

    # --- CLIP 계열 (CLS/EOS Token 사용) ---
    {"name": "OpenAI (Base)", "id": "openai/clip-vit-base-patch32", "type": "clip"},
    {"name": "OpenAI (Large)", "id": "openai/clip-vit-large-patch14", "type": "clip"},
    {"name": "KoCLIP (Base)", "id": "koclip/koclip-base-pt", "type": "clip"},
]

TEST_QUERIES = [
    # 1. [상황/장소] TPO에 맞는 옷 찾기
    "출근할 때 입기 좋은 깔끔한 오피스룩 셔츠",  # (셔츠/블라우스, 슬랙스 타겟)
    "결혼식 하객룩으로 입을 만한 단정한 바지",   # (슈트팬츠/슬랙스 타겟)
    "운동할 때 입기 편한 트레이닝 바지",         # (기타하의 타겟)
    "데이트할 때 입기 좋은 여리여리한 블라우스",  # (셔츠/블라우스 타겟)

    # 2. [스타일/무드] 특정 분위기 연출
    "힙한 스트릿 무드의 나일론 자켓",            # (나일론 코치재킷 타겟)
    "빈티지한 느낌의 와이드 청바지",             # (데님 팬츠 타겟)
    "프레피룩에 어울리는 니트 조끼",             # (베스트 타겟)
    "단정하고 클래식한 카라 티셔츠",             # (피케 카라티셔츠 타겟)

    # 3. [계절/기능] 날씨와 소재 중심
    "찬바람 막아주는 가벼운 아우터",             # (나일론 코치재킷 타겟)
    "겨울에 입기 좋은 두툼하고 따뜻한 니트",      # (니트/스웨터 타겟)
    "여름에 시원하게 입기 좋은 반팔 카라티",      # (피케 카라티셔츠 타겟)
    "간절기에 셔츠 위에 겹쳐 입기 좋은 조끼",     # (베스트 타겟)

    # 4. [디테일/핏] 구체적인 생김새 묘사
    "체형 커버가 되는 넉넉한 핏의 고무줄 바지",   # (기타하의/와이드팬츠 타겟)
    "귀여운 그래픽이 들어간 긴팔 티셔츠",         # (긴소매티셔츠 타겟)
    "다리가 길어 보이는 부츠컷 슬랙스",           # (슈트팬츠/슬랙스 타겟)
    "체크 무늬가 들어간 캐주얼한 셔츠",           # (셔츠/블라우스 타겟)
]
OUTPUT_FILE = "evaluation_results.json"

# ==========================================
# 1. 모델 로드 함수 (범용성 강화)
# ==========================================
def load_text_encoder(config):
    model_type = config['type']
    model_id = config['id']
    
    print(f"   Running Model: {model_id} ({model_type})...")

    if model_type == "sbert":
        return SentenceTransformer(model_id)
    
    elif model_type == "bert":
        # 일반적인 BERT/RoBERTa 모델
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id)
        return model, tokenizer
    
    elif model_type == "clip":
        # OpenAI CLIP 또는 KoCLIP (텍스트 모델 강제 로드)
        if "koclip" in model_id:
            # KoCLIP은 내부적으로 RoBERTa를 사용하므로 AutoModel로 로드해야 함!
            # (CLIPTextModel을 쓰면 가중치가 증발함)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModel.from_pretrained(model_id)
            # 텍스트 모델만 추출하여 'pixel_values' 오류 방지
            if hasattr(model, "text_model"):
                model = model.text_model
        else:
            # OpenAI CLIP 등 표준 CLIP 모델
            tokenizer = CLIPTokenizer.from_pretrained(model_id)
            model = CLIPTextModel.from_pretrained(model_id)
        return model, tokenizer
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# ==========================================
# 2. 임베딩 생성 함수 (모델 타입별 분기)
# ==========================================
def get_embeddings(texts, model, tokenizer, model_type):
    # S-BERT는 내부적으로 처리하므로 여기로 오지 않음 (메인 로직에서 분기)
    
    # 모델 타입에 따라 최대 길이(max_length) 다르게 설정
    if model_type == "clip":
        max_len = 77  # OpenAI CLIP, KoCLIP 등은 77이 한계
    else:
        max_len = 128 # BERT, S-BERT 등은 128~512까지 가능 (속도를 위해 128 유지)

    # 1. 토크나이징
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=max_len)
    
    # OpenAI CLIP만 token_type_ids를 제거 (KoCLIP은 RoBERTa라 필요할 수 있음)
    if hasattr(model.config, 'model_type') and model.config.model_type == 'clip_text_model':
        if 'token_type_ids' in inputs:
            del inputs['token_type_ids']

    # 2. 모델 추론
    with torch.no_grad():
        outputs = model(**inputs)

    # 3. 임베딩 추출 전략
    if model_type == "clip":
        # [CLIP] pooler_output 혹은 last_hidden_state의 첫 번째 토큰 사용
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            embeddings = outputs.pooler_output
        else:
            embeddings = outputs.last_hidden_state[:, 0, :]
            
    elif model_type == "bert":
        # [BERT] Mean Pooling
        token_embeddings = outputs.last_hidden_state
        attention_mask = inputs['attention_mask']
        
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embeddings = sum_embeddings / sum_mask
        
    return embeddings.cpu().numpy()

def l2_normalize(vectors):
    if vectors.ndim == 1: vectors = vectors.reshape(1, -1)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0 
    return vectors / norms

# ==========================================
# 메인 실행 로직
# ==========================================
if __name__ == "__main__":
    
    # 데이터 로드
    try:
        with open('image.json', 'r', encoding='utf-8') as f: data = json.load(f)
    except FileNotFoundError: print("image.json not found"); exit()

    descriptions = [item['description'] for item in data]
    final_results = {}

    print(f"총 {len(MODEL_CONFIGS)}개의 모델을 평가합니다.")

    for config in MODEL_CONFIGS:
        model_name = config['name']
        model_type = config['type']
        
        try:
            # 모델 로드
            encoder = load_text_encoder(config)
            
            # 인덱싱 시작
            start_time = time.time()
            
            if model_type == "sbert":
                # S-BERT는 encode 메서드 바로 사용
                embeddings = encoder.encode(descriptions, convert_to_numpy=True)
            else:
                # BERT/CLIP은 get_embeddings 함수 사용
                model, tokenizer = encoder
                embeddings = get_embeddings(descriptions, model, tokenizer, model_type)
            
            indexing_time = time.time() - start_time
            
            # FAISS 인덱싱
            embeddings = l2_normalize(embeddings)
            D = embeddings.shape[1]
            index = faiss.IndexFlatL2(D)
            index.add(embeddings)
            
            # 쿼리 테스트
            latency_list = []
            search_examples = []
            
            for query in TEST_QUERIES:
                q_start = time.time()
                
                if model_type == "sbert":
                    q_vec = encoder.encode([query], convert_to_numpy=True)
                else:
                    q_vec = get_embeddings([query], model, tokenizer, model_type)
                
                q_vec = l2_normalize(q_vec)
                dists, idxs = index.search(q_vec, 5)
                
                latency_list.append((time.time() - q_start) * 1000)
                
                top_results = []
                for i, idx in enumerate(idxs[0]):
                    if idx < len(data):
                        item = data[idx]
                        top_results.append({
                            "rank": i + 1,
                            "product_name": item['product_name'],
                            "score": float(dists[0][i]),
                            "image": item['image_filename']
                        })
                search_examples.append({"query": query, "results": top_results})

            # 결과 저장
            final_results[model_name] = {
                "type": model_type,
                "indexing_time_sec": round(indexing_time, 2),
                "avg_latency_ms": round(np.mean(latency_list), 2),
                "search_examples": search_examples
            }
            print(f"   ✅ 완료! (Latency: {np.mean(latency_list):.2f}ms)\n")

        except Exception as e:
            print(f"   ❌ 실패 ({model_name}): {e}\n")

    # 결과 파일 저장
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)
    print(f"✨ 모든 평가 완료! '{OUTPUT_FILE}' 저장됨.")