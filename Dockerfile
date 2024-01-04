# 공식 Python 이미지 사용
FROM python:3.9-slim-buster

# 작업 디렉토리 설정
WORKDIR /app

# 필요한 시스템 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing \
        git \
        gcc \
        g++ \
        libgl1-mesa-glx \
        libglib2.0-0 \
        && rm -rf /var/lib/apt/lists/*d

# requirements.txt 파일 복사
COPY requirements.txt /app/

# 필요한 Python 라이브러리 설치
RUN pip install -r requirements.txt

# Detectron2 설치
RUN pip install 'git+https://github.com/facebookresearch/detectron2.git'

# 애플리케이션 코드 복사
COPY . /app

# 컨테이너의 8000 포트에 접근
EXPOSE 8000

# 컨테이너가 시작될 때 실행할 명령어 정의
CMD ["uvicorn", "KNEE_server:app", "--host", "0.0.0.0", "--port", "8000"]
