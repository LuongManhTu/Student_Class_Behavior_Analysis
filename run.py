import argparse
from students_detector import DinoDetector, YoloDetector
from behavior_classifier import Classifier
from deep_sort_realtime.deepsort_tracker import DeepSort
import os
import numpy as np
import cv2
import time
import torch
import gradio as gr
from PIL import Image
import groundingdino.datasets.transforms as T
from typing import Tuple
import torch.nn.functional as F
import torchvision.transforms as transforms
import pandas as pd
import torch.nn as nn
from helper import denormalize_boxes, load_image_from_frame, post_process
import plotly.graph_objects as go

TEXT_PROMPT = "single people"
BOX_THRESHOLD = 0.4
TEXT_THRESHOLD = 0.3

def process_video(input_path, output_path, detector, classifier, tracker):
    tracker = DeepSort(max_age=50)
    video_cap = cv2.VideoCapture(input_path)

    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    # output_path = "processed_video.mp4"
    writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    track_colors = {}
    action_log = {}
    WHITE = (255, 255, 255)

    while True:
        ret, frame = video_cap.read()
        if not ret:
            break

        results = detector(frame)
        
        tracks = tracker.update_tracks(results, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()

            if track_id not in track_colors:
                track_colors[track_id] = tuple(np.random.randint(0, 255, size=3).tolist())

            color = track_colors[track_id]
            xmin, ymin, xmax, ymax = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
            cropped_frame = frame[ymin:ymax, xmin:xmax]

            # Classify action
            if cropped_frame.size != 0:
                action, score, action_id = classifier.forward(Image.fromarray(cropped_frame))

                if track_id not in action_log:
                    action_log[track_id] = []
                action_log[track_id].append(action_id)

                cv2.putText(frame, f"{track_id}: {action}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

        writer.write(frame)

    video_cap.release()
    writer.release()
    # Convert action_log to Pandas DataFrame

    maxlen = max(len(actions) for actions in action_log.values())
    filtered_action_log = {key: actions for key, actions in action_log.items() if len(actions) >= 1.0 * maxlen}

    # Tạo DataFrame từ action_log đã lọc
    action_log_df = pd.DataFrame({
        "Track ID": list(action_log.keys()),
        "Actions": [','.join(map(str, actions)) for actions in action_log.values()]
    })

    return output_path, action_log_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SCB Analysis")
    
    parser.add_argument('--detector', type=str, default="YOLO", choices=["DINO", "YOLO"])
    parser.add_argument('--device', type=str, default="cuda", choices=["cuda", "cpu"])
    
    parser.add_argument('--dino_config_path', type=str, default="./groundingdino/config/GroundingDINO_SwinT_OGC.py")
    parser.add_argument('--dino_weight_path', type=str, default="./weights/groundingdino_swint_ogc.pth")
    parser.add_argument('--yolo_weight_path', type=str, default="./weights/yolov8-best_7-1.pt")
    parser.add_argument('--resnet_weight_path', type=str, default="./weights/resnet_fit.pth")
    
    args = parser.parse_args()

    if args.detector == "DINO":
        detector = DinoDetector(args.dino_config_path, args.dino_weight_path, args.device)
        print(">>>> Loaded Detector: Grounding Dino!")
    else:
        detector = YoloDetector(args.yolo_weight_path, args.device)
        print(">>>> Loaded Detector: YOLOv8!")

    classifier = Classifier(args.resnet_weight_path, args.device)
    print(">>>> Loaded Classifier: Resnet!")

    if args.device == "cuda":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    tracker = DeepSort(max_age=50)

    input_path = "./test_videos/first.mp4"
    output_path = "./results/first_vis.mp4"

    output_video, action_log_df = process_video(input_path, output_path, detector, classifier, tracker)

    action_log_df.to_csv('./results/action_log.csv', index=False)

    df = action_log_df

    # Đổi tên cột cho thống nhất với logic cũ
    df = df.rename(columns={"Track ID": "student_id", "Time": "frame", "Action": "action"})

    # Đảm bảo các cột có kiểu dữ liệu đúng
    df["student_id"] = df["student_id"].astype(int)
    df["frame"] = df["frame"].astype(int)
    df["action"] = df["action"].astype(int)

    # Tạo danh sách students từ df
    num_frames = df['frame'].nunique()  # Số lượng frame duy nhất
    students = []
    for student_id, group in df.groupby('student_id'):
        actions = group.sort_values('frame')['action'].tolist()
        students.append({"student_id": student_id, "actions": actions})


    action_dict = {
        0: "Raising hand",
        1: "Reading",
        2: "Sleeping",
        3: "Using phone",
        4: "Writing"
    }

    action_labels = ['Raising hand', 'Reading', 'Sleeping', 'Using phone', 'Writing']
    action_colors = ['green', 'lightgreen', 'red', 'orange', 'gold'] 

    weights = {
        0: 5,  # Raising hand
        1: 2,  # Reading
        2: -5,  # Sleeping
        3: -3,  # Using phone
        4: 3   # Writing
    }
    df["weight"] = df["action"].map(weights)


    # Tính điểm trung bình cho từng frame
    average_scores = (
        df.groupby("frame")["weight"]
        .mean()
        .reset_index()
        .rename(columns={"weight": "average_score"})
    )

    # Tính điểm trung bình và mức độ tập trung của sinh viên
    student_avg_scores = df.groupby("student_id")["weight"].mean().reset_index()
    student_avg_scores["concentration_level"] = student_avg_scores["weight"].apply(
        lambda x: "High" if x > 2.5 else ("Normal" if x >= 0 else "Low")
    )

    # Tính điểm trung bình của lớp
    class_avg_score = df["weight"].mean()
    class_concentration_level = "High" if class_avg_score > 2.5 else ("Normal" if class_avg_score >= 0 else "Low")


    def prepare_grouped_timeline():
        # Tạo cột màu và nhãn cho từng hành động
        action_dict = {
            0: "Raising hand",
            1: "Reading",
            2: "Sleeping",
            3: "Using phone",
            4: "Writing"
        }
        action_colors = ['green', 'lightgreen', 'red', 'orange', 'gold']
        
        df["action_color"] = df["action"].map(dict(zip(range(len(action_colors)), action_colors)))
        df["action_label"] = df["action"].map(action_dict)
        
        # Nhóm các hành động liên tiếp giống nhau
        df["group"] = (df["action"] != df["action"].shift()).cumsum()
        
        grouped = (
            df.groupby(["student_id", "group", "action", "action_color", "action_label"], as_index=False)
            .agg(start_frame=("frame", "min"), end_frame=("frame", "max"))
        )
        
        return grouped


    def create_grouped_timeline():
        grouped_data = prepare_grouped_timeline()

        # Lấy danh sách các ID sinh viên thực tế và ánh xạ thành chỉ số liên tiếp
        unique_ids = sorted(grouped_data["student_id"].unique())
        id_mapping = {student_id: idx + 1 for idx, student_id in enumerate(unique_ids)}

        # Thay thế `student_id` gốc bằng chỉ số liên tiếp
        grouped_data["mapped_id"] = grouped_data["student_id"].map(id_mapping)

        # Định nghĩa danh sách nhãn và màu sắc cố định
        actions_colors = {
            "reading": "lightgreen",
            "writing": "gold",
            "raising hand": "green",
            "using phone": "orange",
            "sleeping": "red",
        }

        fig = go.Figure()

        # Thêm các nhãn cố định vào chú thích
        for action, color in actions_colors.items():
            fig.add_trace(
                go.Scatter(
                    x=[None],  # Không hiển thị trên biểu đồ chính
                    y=[None],
                    mode="lines",
                    line=dict(color=color, width=10),
                    name=action,
                    showlegend=True,
                )
            )

        # Duyệt qua từng sinh viên và từng hành động để vẽ dữ liệu thực tế
        for mapped_id, student_data in grouped_data.groupby("mapped_id"):
            for _, row in student_data.iterrows():
                fig.add_trace(
                    go.Scatter(
                        x=[row["start_frame"], row["end_frame"]],  # Khung hình bắt đầu và kết thúc
                        y=[mapped_id, mapped_id],  # Dòng của sinh viên (liên tục)
                        mode="lines",
                        line=dict(color=row["action_color"], width=10),
                        name=row["action_label"],
                        showlegend=False,  # Không cần lặp lại nhãn
                    )
                )

        fig.update_layout(
            title="Action Timeline for Each Student",
            xaxis_title="Frame (Time)",
            yaxis_title="Student ID",
            height=600,
            legend_title="Actions",
            yaxis=dict(
                tickmode="array",  # Hiển thị các giá trị cụ thể
                tickvals=list(id_mapping.values()),  # Hiển thị ID ánh xạ liên tiếp
                ticktext=list(id_mapping.keys()),  # Văn bản hiển thị là ID gốc
            ),
            xaxis=dict(
                tickmode="linear",  # Hiển thị trục X theo dạng tuyến tính
                tick0=0,  # Bắt đầu từ 0
                dtick=50,  # Khoảng cách giữa các tick là 50 frame
            ),
        )

        return fig


    # Hàm tạo timeline điểm trung bình
    def create_average_score_timeline():
        fig = go.Figure(
            data = go.Scatter(
                x = average_scores["frame"],
                y = average_scores["average_score"],
                mode = "lines + markers",
                line = dict(color="blue"),
                marker = dict(size=5)
            )
        )
        fig.update_layout(
            title = "Average Class Score Over Time",
            xaxis_title = "Frame",
            yaxis_title = "Average Score",
            xaxis = dict(tickvals=np.arange(0, num_frames, 50))
        )
        return fig

    # Tính tỷ lệ của các hành động
    action_counts = df["action"].value_counts().reset_index()
    action_counts.columns = ["action", "count"]

    def create_action_pie_chart():
        # Sắp xếp lại action_counts theo thứ tự của action_labels
        sorted_actions = pd.DataFrame({
            "action": range(len(action_labels)),  # Tạo thứ tự dựa trên action_labels
            "label": action_labels,
            "color": action_colors
        })
        action_counts_sorted = sorted_actions.merge(
            action_counts, left_on="action", right_on="action", how="left"
        ).fillna(0)  # Điền giá trị 0 cho các hành động không xuất hiện

        # Tạo biểu đồ
        fig = go.Figure(
            data=go.Pie(
                labels=action_counts_sorted["label"],
                values=action_counts_sorted["count"],
                marker=dict(colors=action_counts_sorted["color"]),
                textinfo="label + percent",
                hoverinfo="label + value"
            )
        )
        fig.update_layout(title="Action Distribution Across All Frames")
        return fig


    # Hàm tạo bảng kết quả sinh viên
    def create_student_summary_table():
        student_table = student_avg_scores.copy()
        student_table["concentration_level"] = student_table["concentration_level"].map(
            {"High": "Tập trung cao", "Normal": "Bình thường", "Low": "Mất tập trung"}
        )
        return student_table

    # Hàm tạo bảng kết quả cho cả lớp
    def create_class_summary():
        return f"""
        **Class Average Score**: {class_avg_score:.2f}
        **Class Concentration Level**: {"Tập trung cao" if class_concentration_level == "High" else "Bình thường" if class_concentration_level == "Normal" else "Mất tập trung"}
        """
    
    # Hàm xử lý video
    def process_video_function(input_video):
        output_video, action_log_df = process_video(input_video, classifier, device)
        # Lưu action_log ra file CSV
        action_log_df.to_csv('./results/action_log.csv', index=False)
        print("Action log saved to ./results/action_log.csv")
        return output_video

    # Hàm để cập nhật visibility của analysis_column sau 5 giây
    def show_analysis_after_delay():
        time.sleep(350)
        return gr.update(visible=True)  # Thay đổi visible thành True

    # Giao diện cho phần xử lý video
    video_interface = gr.Interface(
        fn=process_video_function,
        inputs=gr.Video(label="Upload a video"),
        outputs=gr.Video(label="Processed video"),
        title="Video Processing with GroundingDINO and DeepSORT",
    )

    # Giao diện tổng hợp
    with gr.Blocks() as demo:
        gr.Markdown("# Student Behavior Analysis via CCTV videos")
        
        # Giao diện xử lý video
        with gr.Column():  # To make video processing section take full width
            # gr.Markdown("## Video Processing")
            video_interface.render()  # Thêm giao diện xử lý video vào đây
        
        # Phân tích hành vi sinh viên
        with gr.Column(visible=False) as analysis_column:  # Đảm bảo phần này cũng chiếm toàn bộ chiều rộng dưới video
            gr.Markdown("## Student Behavior Analysis")
            
            with gr.Tab("Timeline of Actions") as timeline_tab:
                gr.Plot(create_grouped_timeline())
            
            with gr.Tab("Average Class Score Over Time") as avg_score_tab:
                gr.Plot(create_average_score_timeline())
        
            with gr.Tab("Action Distribution (Pie Chart)") as pie_chart_tab:
                gr.Plot(create_action_pie_chart())
        
            with gr.Tab("Student Performance Table") as student_table_tab:
                gr.Markdown("### Student Performance")
                student_table_data = create_student_summary_table()
                gr.DataFrame(
                    value=student_table_data,
                    headers=["Student ID", "Average Score", "Concentration Level"]
                )
                gr.Markdown("### Class Performance")
                gr.Markdown(create_class_summary())
        
        # Đặt phần phân tích hành vi sinh viên hiện ra sau 5 giây tự động
        demo.load(show_analysis_after_delay, outputs=analysis_column)

    # Chạy ứng dụng Gradio
    demo.launch(debug=True, share=True)