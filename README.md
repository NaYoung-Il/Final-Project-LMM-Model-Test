/images : 이미지(데이터셋) 470개

image.csv : 470개 이미지 데이터셋 (엑셀)

image.json : image.csv를 JSON 형태로 변환

csv_to_json.py : image.csv를 image.json으로 변환하는 파일


1. Docker 없이 실행 (Python 가상환경)

- 가상환경 설치 : Ctrl + Shift + P -> Python:Select Interpreter -> Create Virtual Environment -> .venv (Python 3.10.0)

- 가상환경 안에서 실행 -> 의존성 설치

pip install -r requirements2.txt

- search_engine.py (모델 평가) 실행 -> evaluation_results.json 생성 (app.py에서 사용)  

python search_engine.py

- app.py 실행 -> streamlit으로 시각화

streamlit run app.py


2. Docker로 실행 (Docker Desktop 설치 필요) -> 오래 걸릴 수 있음(특히 학원컴)

도커 이미지 이름(lmm-search)은 원하시는 대로 정하셔도 됩니다.

마지막의 점(.)은 현재 폴더를 의미합니다. (필수!)

docker build -t lmm-search .

-p 8501:8501 => 내 컴퓨터의 8501 포트와 도커의 8501 포트를 연결

docker run -p 8501:8501 lmm-search

인터넷 브라우저를 켜고 **http://localhost:8501**에 접속하면 Streamlit 앱이 실행됨.
