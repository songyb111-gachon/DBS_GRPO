import re
import tkinter as tk
from tkinter import simpledialog, filedialog


def extract_dataset_steps(file_path, channel, image_size):
    results = []
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

        # 데이터셋별로 나누기
        dataset_blocks = re.split(r"\[Episode Start\] Currently using dataset file: \((.+?)\), Episode count: \d+", content)

        # 첫 번째 항목은 데이터셋 이전의 로그 내용이므로 제외
        datasets = dataset_blocks[1:]

        for i in range(0, len(datasets), 2):
            dataset_file = datasets[i].strip()
            dataset_content = datasets[i + 1]
            dataset_results = [f"Dataset: {dataset_file}\n"]

            # 일반 스텝 데이터 추출
            step_matches = re.finditer(
                r"Step: (\d+)\s+"
                r"PSNR Before: [\d.]+\s+\|\s+PSNR After: [\d.]+\s+\|\s+Change: [\d.e+-]+\s+\|\s+Diff: ([\d.e+-]+)\s+"
                r"Success Ratio: ([\d.e+-]+)\s+\|\s+Flip Count: (\d+).*?"
                r"Time taken for this data: ([\d.]+) seconds",
                dataset_content,
                re.DOTALL,
            )

            for match in step_matches:
                step = int(match.group(1))
                psnr_diff = float(match.group(2))  # 일반 스텝에서는 Diff를 그대로 사용
                success_ratio = float(match.group(3))
                flip_count = int(match.group(4))
                time_taken = float(match.group(5))

                dataset_results.append(
                    f"PSNR diff: {psnr_diff:.6f}\n"
                    f"Step: {step}\n"
                    f"Time: {time_taken:.2f}s\n"
                    f"Success Ratio: {success_ratio:.6f}\n"
                    f"Flip Count: {flip_count}\n"
                )

            # 마지막 스텝 데이터 추출
            final_step = channel * image_size * image_size
            final_step_match = re.search(
                rf"Step: {final_step}\s+"
                r"PSNR Before: [\d.]+\s+\|\s+PSNR After: [\d.]+\s+\|\s+Change: [\d.e+-]+\s+\|\s+Diff: ([\d.e+-]+).*?"
                r"Success Ratio: ([\d.e+-]+)\s+\|\s+Flip Count: (\d+).*?"
                r"Time taken for this data: ([\d.]+) seconds",
                dataset_content,
                re.DOTALL,
            )
            if final_step_match:
                psnr_diff = float(final_step_match.group(1))  # 마지막 스텝도 Diff(총 개선량). Change 는 마지막 임계 통과 시점의 단일 플립 변화량이라 총 개선량이 아니다
                success_ratio = float(final_step_match.group(2))
                flip_count = int(final_step_match.group(3))
                time_taken = float(final_step_match.group(4))

                dataset_results.append(
                    f"PSNR diff: {psnr_diff:.6f}\n"
                    f"Step: {final_step}\n"
                    f"Time: {time_taken:.2f}s\n"
                    f"Success Ratio: {success_ratio:.6f}\n"
                    f"Flip Count: {flip_count}\n"
                )

            results.append("\n".join(dataset_results))

    return results


def open_file_and_extract():
    # GUI를 통해 사용자 입력 받기
    channel = simpledialog.askinteger("Input", "Enter the number of channels (e.g., 3):", minvalue=1)
    image_size = simpledialog.askinteger("Input", "Enter the image size (e.g., 512):", minvalue=1)
    if not channel or not image_size:
        print("Channel or image size input cancelled.")
        return

    # 파일 선택 GUI 열기
    file_path = filedialog.askopenfilename(
        title="Select Log File",
        filetypes=[("Log Files", "*.log"), ("All Files", "*.*")]
    )
    if not file_path:
        print("No file selected.")
        return

    # 데이터셋별 스텝 데이터 추출
    results = extract_dataset_steps(file_path, channel, image_size)

    # 결과 GUI 창에 표시
    result_window = tk.Toplevel()
    result_window.title("Step Data Results")
    text_widget = tk.Text(result_window, wrap="word", width=80, height=20)
    text_widget.pack(expand=True, fill="both")
    text_widget.insert("1.0", "\n\n".join(results))

    # 결과 저장 옵션
    output_file = filedialog.asksaveasfilename(
        title="Save Results As",
        defaultextension=".txt",
        filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
    )
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            for result in results:
                f.write(result + "\n\n")
        print(f"Results saved to {output_file}")


# GUI 설정
root = tk.Tk()
root.title("Dataset Step Extractor")

# 버튼 추가
select_button = tk.Button(root, text="Select Log File", command=open_file_and_extract, width=30)
select_button.pack(pady=20)

# 메인 루프 실행
root.mainloop()