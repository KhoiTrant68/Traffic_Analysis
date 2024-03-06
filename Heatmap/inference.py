import cv2
import argparse

import supervision as sv
from ultralytics import YOLO


def heatmap_and_track(
    model_id: str,
    video_path: str,
    save_path: str,
    confidence_threshold: float = 0.35,
    iou_threshold: float = 0.5,
    heatmap_alpha: float = 0.5,
    radius: int = 25,
    track_threshold: float = 0.35,
    track_seconds: int = 5,
    match_threshold: float = 0.99,
) -> None:
    ### Instantiate model
    model = YOLO(model_id)

    ### Heatmap config
    heat_map_annotator = sv.HeatMapAnnotator(
        position=sv.Position.BOTTOM_CENTER,
        opacity=heatmap_alpha,
        radius=radius,
        kernel_size=25,
        top_hue=0,
        low_hue=125,
    )

    ### Annotation config
    label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)

    ### Get the video fps
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    ### Tracker config
    byte_tracker = sv.ByteTrack(
        track_thresh=track_threshold,
        track_buffer=track_seconds * fps,
        match_thresh=match_threshold,
        frame_rate=fps,
    )

    ### Video config
    video_info = sv.VideoInfo.from_video_path(video_path=video_path)
    frames_generator = sv.get_video_frames_generator(source_path=video_path, stride=1)

    ### Detect, track, annotate, save
    with sv.VideoSink(target_path=save_path, video_info=video_info) as sink:
        for frame in frames_generator:
            result = model(
                source=frame,
                classes=[0],  # only person class
                conf=confidence_threshold,
                iou=iou_threshold,
                # show_conf = True,
                # save_txt = True,
                # save_conf = True,
                # save = True,
                device=None,  # use None = CPU, 0 = single GPU, or [0,1] = dual GPU
            )[0]

            detections = sv.Detections.from_ultralytics(result)  # get detections

            detections = byte_tracker.update_with_detections(
                detections
            )  # update tracker

            ### draw heatmap
            annotated_frame = heat_map_annotator.annotate(
                scene=frame.copy(), detections=detections
            )

            ### draw other attributes from `detections` object
            labels = [
                f"#{tracker_id}"
                for class_id, tracker_id in zip(
                    detections.class_id, detections.tracker_id
                )
            ]

            label_annotator.annotate(
                scene=annotated_frame, detections=detections, labels=labels
            )

            sink.write_frame(frame=annotated_frame)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Heatmap and Tracking with Supervision"
    )
    parser.add_argument(
        "--model_id",
        required=True,
        help="Type of YOLO model",
        type=str,
    )
    parser.add_argument(
        "--video_path",
        help="Path to the source video file",
        type=str,
    )
    parser.add_argument(
        "--save_path",
        default="output.mp4",
        help="Path to the target video file (output)",
        type=str,
    )
    parser.add_argument(
        "--confidence_threshold",
        default=0.35,
        help="Confidence threshold for the model",
        type=float,
    )
    parser.add_argument(
        "--iou_threshold",
        default=0.5,
        help="IOU threshold for the model",
        type=float,
    )
    parser.add_argument(
        "--heatmap_alpha",
        default=0.5,
        help="Opacity of the overlay mask, between 0 and 1",
        type=float,
    )
    parser.add_argument(
        "--radius",
        default=25,
        help="Radius of the heat circle",
        type=float,
    )
    parser.add_argument(
        "--track_threshold",
        default=0.35,
        help="Detection confidence threshold for track activation",
        type=float,
    )
    parser.add_argument(
        "--track_seconds",
        default=5,
        help="Number of seconds to buffer when a track is lost",
        type=int,
    )
    parser.add_argument(
        "--match_threshold",
        default=0.99,
        help="Threshold for matching tracks with detections",
        type=float,
    )

    args = parser.parse_args()

    heatmap_and_track(
        model_id=args.model_id,
        video_path=args.video_path,
        save_path=args.save_path,
        confidence_threshold=args.confidence_threshold,
        iou_threshold=args.iou_threshold,
        heatmap_alpha=args.heatmap_alpha,
        radius=args.radius,
        track_threshold=args.track_threshold,
        track_seconds=args.track_seconds,
        match_threshold=args.match_threshold,
    )
