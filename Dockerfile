# 1. 베이스 이미지 (가벼운 파이썬 3.10 버전 사용)
FROM python:3.10-slim

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. 필수 시스템 패키지 설치 (혹시 모를 CV2 등의 의존성 대비)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. 의존성 파일 복사 및 설치
# (이 단계를 먼저 수행해야 코드 수정 시 캐시를 활용해 빌드 속도가 빨라짐)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 현재 폴더의 모든 파일을 컨테이너의 /app으로 복사
COPY . .

# 6. (중요) Streamlit이 사용하는 포트 개방
EXPOSE 8501

# 7. 컨테이너 실행 시 작동할 명령어
# 검색 결과 파일(evaluation_results.json)이 없으면 생성 후 실행하는 스크립트
# (간단하게 바로 app.py를 실행하도록 설정함. 필요시 search_engine.py 먼저 실행해야 함)
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]