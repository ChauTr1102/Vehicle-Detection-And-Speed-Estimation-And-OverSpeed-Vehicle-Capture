import cv2


import numpy as np
import supervision as sv

from tqdm import tqdm
from ultralytics import YOLO
from supervision.assets import VideoAssets, download_assets
from collections import defaultdict, deque




SOURCE_VIDEO_PATH = "clip1.mp4"
# SOURCE_VIDEO_PATH = cv2.VideoCapture(0)
# SOURCE_VIDEO_PATH.set(cv2.CAP_PROP_FRAME_WIDTH,720)
# SOURCE_VIDEO_PATH.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
TARGET_VIDEO_PATH = "vehicles-result.mp4"
CONFIDENCE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5
# MODEL_NAME = "visdrone_yolov8s.pt"
MODEL_RESOLUTION = 1720


frame_generator = sv.get_video_frames_generator(source_path=SOURCE_VIDEO_PATH)
frame_iterator = iter(frame_generator)
frame = next(frame_iterator)
annotated_frame = frame.copy()



# Global variables to store the clicked points
points = []
point_store = []
def Capture_Event(event, x, y, flags, params):
    global points

    # If the left mouse button is pressed
    if event == cv2.EVENT_LBUTTONDOWN:
        # Append the clicked point to the list
        points.append((x, y))

        # Print the coordinates
        print(f"({x}, {y})")
        point_store.append([x,y])

        # Draw lines between consecutive points
        if len(points) > 1:
            cv2.line(frame, points[-2], points[-1], (0, 255, 0), 2)
        if len(points) == 4:
            cv2.line(frame, points[0], points[-1], (0, 255, 0), 2)
        # Display the updated image
        cv2.imshow('image', frame)

if __name__ == "__main__":
    # Read the image

    cv2.imshow('image', frame)

    # Set the mouse callback function
    cv2.setMouseCallback('image', Capture_Event)

    # Wait for the user to press any key to exit
    cv2.waitKey(0)

    # Destroy all the windows
    cv2.destroyAllWindows()


# SOURCE = np.array([
#     [point_store[0]],
#     [point_store[1]],
#     [point_store[2]],
#     [point_store[3]]
# ])
SOURCE = np.array([
    [521, 212],
    [691, 209],
    [1142, 609],
    [53, 616]
])
TARGET_WIDTH = 6
TARGET_HEIGHT = 24

TARGET = np.array([
    [0, 0],
    [TARGET_WIDTH - 1, 0],
    [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
    [0, TARGET_HEIGHT - 1],
])



annotated_frame = sv.draw_polygon(scene=annotated_frame, polygon=SOURCE, color=sv.Color.red(), thickness=4)
sv.plot_image(annotated_frame)


import matplotlib.pyplot as plt
plt.imshow(annotated_frame)

class ViewTransformer:

    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)


class ViewTransformer:

    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)


view_transformer = ViewTransformer(source=SOURCE, target=TARGET)
model = YOLO('visdrone_yolov8s.pt')
names = model.model.names
video_info = sv.VideoInfo.from_video_path(video_path=SOURCE_VIDEO_PATH)
frame_generator = sv.get_video_frames_generator(source_path=SOURCE_VIDEO_PATH)

# tracer initiation
byte_track = sv.ByteTrack(
    frame_rate=video_info.fps, track_thresh=CONFIDENCE_THRESHOLD
)

# annotators configuration
thickness = sv.calculate_dynamic_line_thickness(
    resolution_wh=video_info.resolution_wh
)
text_scale = sv.calculate_dynamic_text_scale(
    resolution_wh=video_info.resolution_wh
)
bounding_box_annotator = sv.BoundingBoxAnnotator(
    thickness=thickness
)
label_annotator = sv.LabelAnnotator(
    text_scale=text_scale,
    text_thickness=thickness,
    text_position=sv.Position.BOTTOM_CENTER
)
trace_annotator = sv.TraceAnnotator(
    thickness=thickness,
    trace_length=video_info.fps * 2,
    position=sv.Position.BOTTOM_CENTER
)

polygon_zone = sv.PolygonZone(
    polygon=SOURCE,
    frame_resolution_wh=video_info.resolution_wh
)

coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))
target_line = False
# open target video

with sv.VideoSink(TARGET_VIDEO_PATH, video_info) as sink:
    over_speed_frame = []
    car_pos = []
    tracker_list = []
    store_car_and_pos = dict()
    car_list_post = []
    # loop over source video frame
    for frame in tqdm(frame_generator, total=video_info.total_frames):

        result = model(frame, imgsz=MODEL_RESOLUTION, verbose=False)[0]
        #clss = result[0].boxes.cls.cpu().tolist()

        for r in result:
          for c in r.boxes.cls:
            probs=names[int(c)]
        #probs = result.probs
        detections = sv.Detections.from_ultralytics(result)

        # filter out detections by class and confidence
        detections = detections[detections.confidence > CONFIDENCE_THRESHOLD]
        detections = detections[detections.class_id != 0]

        # filter out detections outside the zone
        detections = detections[polygon_zone.trigger(detections)]

        # refine detections using non-max suppression
        detections = detections.with_nms(IOU_THRESHOLD)

        # pass detection through the tracker
        detections = byte_track.update_with_detections(detections=detections)

        points = detections.get_anchors_coordinates(
            anchor=sv.Position.BOTTOM_CENTER
        )

        # calculate the detections position inside the target RoI
        points = view_transformer.transform_points(points=points).astype(int)

        # store detections position
        for tracker_id, [_, y] in zip(detections.tracker_id, points):
            coordinates[tracker_id].append(y)
        # format labels
        labels = []
        #cls_name=names[(clss)]
        a =   detections.tracker_id

        for tracker_id in detections.tracker_id:
            # print(detections.tracker_id)

            if len(coordinates[tracker_id]) < video_info.fps / 2:
                labels.append(f"#{tracker_id}")
            else:


                coordinate_start = coordinates[tracker_id][-1]
                coordinate_end = coordinates[tracker_id][0]
                distance = abs(coordinate_start - coordinate_end)
                time = len(coordinates[tracker_id]) / video_info.fps
                speed = distance / time * 3.6

                if int(speed) >50:
                    a = detections.tracker_id
                    labels.append(f"{probs}:#{tracker_id} {int(speed)} km/h is overspeed")
                    if tracker_id not in tracker_list:
                      tracker_list.append(tracker_id)
                      over_speed_frame.append(frame)
                      pos = np.where(a==tracker_id)[0][0]
                      car_list_post.append(detections.xyxy[pos])
                      car_pos.append(pos)
                      # print(detections.xyxy[pos])


                    # cropped_image = sv.crop_image(image=frame, xyxy=detections.xyxy[pos])
                    # sv.plot_image(cropped_image)
                else:
                    labels.append(f"{probs}:#{tracker_id} {int(speed)} km/h")
        # annotate frame
        annotated_frame = frame.copy()
        annotated_frame = trace_annotator.annotate(
            scene=annotated_frame, detections=detections
        )
        annotated_frame = bounding_box_annotator.annotate(
            scene=annotated_frame, detections=detections
        )
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, detections=detections, labels=labels
        )
        sink.write_frame(annotated_frame)

with sv.ImageSink(target_dir_path='D:\DAP391m',
                              overwrite=True) as sink:
                count = 0
for f in over_speed_frame:
  cropped_image = sv.crop_image(image=f, xyxy=car_list_post[count])
  count +=1
  sv.plot_image(cropped_image)
  sink.save_image(image=cropped_image)

