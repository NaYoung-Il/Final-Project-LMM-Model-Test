import pandas as pd

# csv 파일 읽어오기
# encoding='cp949'를 추가하여 한글 깨짐 방지
df = pd.read_csv('image.csv', encoding='cp949')

# JSON으로 변환
# orient='records': 리스트 안에 딕셔너리가 들어가는 형태 [{}, {}, ...]
# force_ascii=False: 한글이 깨지지 않고 그대로 저장되게 함
json_output_path = 'image.json'
df.to_json(json_output_path, orient='records', force_ascii=False, indent=4)

# (선택) JSON 파일 읽어서 출력 -> 테스트용
with open(json_output_path, 'r', encoding='utf-8') as f:
    json_content = f.read()

print(f"JSON file created at: {json_output_path}")
print("First few characters of JSON content:")
print(json_content[:500])