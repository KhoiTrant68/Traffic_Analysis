import os
import cv2
import json
import argparse
import numpy as np

font = cv2.FONT_HERSHEY_SIMPLEX


def find_vertical_zones(image):
    """Return vertical zones (2 zones)"""
    h, w = image.shape[:2]
    polygon_1 = [[0, 0], [w // 2, 0], [w // 2, h], [0, h]]
    polygon_2 = [[w // 2, 0], [w, 0], [w, h], [w // 2, h]]
    polygons = [polygon_1, polygon_2]
    return polygons


def find_horizontal_zones(image):
    """Return horizontal zones (2zones)"""
    h, w = image.shape[:2]
    polygon_1 = [[0, 0], [w, 0], [w, h // 2], [0, h // 2]]
    polygon_2 = [[0, h // 2], [w, h // 2], [w, h], [0, h]]
    polygons = [polygon_1, polygon_2]
    return polygons


def find_quarter_zones(image):
    """Return quarter zones (4 zones)"""
    h, w = image.shape[:2]
    polygon_1 = [[0, 0], [w // 2, 0], [w // 2, h // 2], [0, h // 2]]
    polygon_2 = [[w // 2, 0], [w, 0], [w, h // 2], [w // 2, h // 2]]
    polygon_3 = [[w // 2, h // 2], [w, h // 2], [w, h], [w // 2, h]]
    polygon_4 = [[0, h // 2], [w // 2, h // 2], [w // 2, h], [0, h]]
    polygons = [
        polygon_1,
        polygon_2,
        polygon_3,
        polygon_4,
    ]
    return polygons


def find_hexagon_zones(image):
    """Return hexagon zones (7 zones)"""
    h, w = image.shape[:2]

    polygon_1 = [[0, 0], [w // 4, h // 4], [0, h // 2]]
    polygon_2 = [[0, 0], [w, 0], [3 * w // 4, h // 4], [w // 4, h // 4]]
    polygon_3 = [[w, 0], [3 * w // 4, h // 4], [w, h // 2]]
    polygon_4 = [[w, h // 2], [3 * w // 4, 3 * h // 4], [w, h]]
    polygon_5 = [[0, h], [w // 4, 3 * h // 4], [3 * w // 4, 3 * h // 4], [w, h]]
    polygon_6 = [[0, h // 2], [w // 4, 3 * h // 4], [0, h]]
    polygon_7 = [
        [w // 4, h // 4],
        [3 * w // 4, h // 4],
        [w, h // 2],
        [3 * w // 4, 3 * h // 4],
        [w // 4, 3 * h // 4],
        [0, h // 2],
    ]

    polygons = [
        polygon_1,
        polygon_2,
        polygon_3,
        polygon_4,
        polygon_5,
        polygon_6,
        polygon_7,
    ]
    return polygons


def visualize_zones(image, polygons):
    """Visualize polygon zones"""
    green = (0, 255, 0)
    for polygon in polygons:
        cv2.polylines(
            image,
            [np.array(polygon, np.int32)],
            isClosed=True,
            color=green,
            thickness=1,
        )
        cv2.imshow("frame", image)
        cv2.waitKey(0)
    cv2.destroyAllWindows()


def write_polygons_file(polygons, save_file):
    """Write results in json file"""
    result_dict = {"polygons": polygons}
    with open(save_file, "w") as outfile:
        json.dump(result_dict, outfile)


def main():
    parser = argparse.ArgumentParser(description="Counting people in zones")
    parser.add_argument(
        "--video_path", required=True, help="Path to the video", type=str
    )
    parser.add_argument("--save_path", help="Path to save json file", type=str)
    parser.add_argument(
        "--type_zone",
        help="Type of zones you want to split. One of ('vertical', 'horizontal', 'quater', 'hexagon')",
        choices=["vertical", "horizontal", "quarter", "hexagon"],
        type=str,
        default="hexagon",
    )
    parser.add_argument("--visualize", help="Visual or not", action="store_true")

    args = parser.parse_args()

    # Argument
    video_path = args.video_path
    print(video_path)
    save_path = args.save_path
    type_zone = args.type_zone


    # Define save path
    name = video_path.split("\\")[-1].split(".")[0]
    save_name = name + "_" + type_zone

    os.makedirs(save_path, exist_ok=True)
    if save_path is not None:
        save_file = os.path.join(save_path, save_name + ".json")
    else:
        save_file = os.path.join(save_name + ".json")

    # Run
    cap = cv2.VideoCapture(video_path)
    _, frame = cap.read()

    if type_zone == "vertical":
        polygons = find_vertical_zones(frame)
    elif type_zone == "horizontal":
        polygons = find_horizontal_zones(frame)
    elif type_zone == "quarter":
        polygons = find_quarter_zones(frame)
    else:
        polygons = find_hexagon_zones(frame)

    if args.visualize:
        visualize_zones(frame, polygons)
    write_polygons_file(polygons, save_file)

    print("Done!")


if __name__ == "__main__":
    main()
