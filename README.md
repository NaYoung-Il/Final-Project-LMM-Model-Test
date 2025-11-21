/images : 이미지(데이터셋) 470개
image.csv : 470개 데이터셋
image.json : image.csv를 JSON 형태로 변환
csv_to_json.py : image.csv를 image.json으로 변환하는 파일

- 가상환경 설치 후 가상환경 안에서 실행 -> 의존성 설치
pip install -r requirements.txt

- search_engine.py 실행 -> 모델 평가
python search_engine.py

evaluation_results.json 생성 -> app.py에서 사용

- app.py 실행 -> streamlit으로 시각화
streamlit run app.py
