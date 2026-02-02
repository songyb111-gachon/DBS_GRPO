import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import os

# 같은 폴더에 있는 TensorBoard 로그 파일의 이름을 자동으로 검색
log_dir = "."  # 현재 폴더|
event_files = [f for f in os.listdir(log_dir) if f.startswith("events.out")]

if not event_files:
    print("No TensorBoard event files found in the directory.")
else:
    file_path = os.path.join(log_dir, event_files[0])  # 첫 번째 로그 파일 선택


    def extract_and_plot(tag, ylabel, title):
        """
        특정 태그에 대해 데이터를 추출하고 그래프를 그리는 함수
        """
        data = []  # 데이터를 태그별로 분리하여 저장
        for event in tf.compat.v1.train.summary_iterator(file_path):

            for value in event.summary.value:
                if value.tag == tag:
                    data.append((event.step, value.simple_value))

        # 데이터를 DataFrame으로 변환
        df = pd.DataFrame(data, columns=["Step", ylabel])

        # 그래프 그리기
        if not df.empty:
            plt.figure(figsize=(10, 6))
            plt.plot(df["Step"], df[ylabel], label=title)
            plt.xlabel("Step")
            plt.ylabel(ylabel)
            plt.title(title)
            plt.legend()
            plt.grid()
            plt.show()
        else:
            print(f"No data found for the tag '{tag}'.")


    # 태그별로 데이터 추출 및 시각화
    extract_and_plot("rollout/ep_rew_mean", "Episode Reward Mean", "Episode Reward Mean over Training Steps")
    extract_and_plot("rollout/ep_len_mean", "Episode Length Mean", "Episode Length Mean over Training Steps")

# 고유 태그 저장
unique_tags = set()

# 로그 파일 읽기
for event in tf.compat.v1.train.summary_iterator(file_path):
    for value in event.summary.value:
        unique_tags.add(value.tag)

# 모든 태그 출력"
print("Unique Tags in the TensorBoard file:")
for tag in unique_tags:
    print(tag)