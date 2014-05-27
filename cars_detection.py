import cv2
from SimpleCV import ndimage, Image, Color
import time

__author__ = 'jeremy'


class VehicleTrackSet(object):
    def __init__(self, track_set):
        # In order to de-dup vehicle, we try to find nearest blob from last frame.
        # If one track set has not be selected for 10 times, we will not consider it.
        self.visit_times = 10
        self.track_set = track_set

    def decrement_visit(self):
        self.visit_times -= 1


# Each vehicle has its own vehicleTrackSet object
vehicle_track_set_list = list()


def nearest_track_set(bounding_box):
    """
    Returns given bounding_box's nearest track_set, x-axis difference, y-axis difference.

    If one track set has not been selected for a long time, we remove it from track_set_list.
    """
    track_set_candidate = dict()

    for vehicle_track_set in vehicle_track_set_list:
        if vehicle_track_set.visit_times > 0:
            track_set = vehicle_track_set.track_set
            diff_x = abs(track_set[-1].x - bounding_box[0])
            diff_y = abs(track_set[-1].y - bounding_box[1])
            #print("Coordinate difference: ({0}, {1}).".format(diff_x, diff_y))
            if diff_x < 40 and diff_y < 40:
                lost_value = diff_x ** 2 + diff_y ** 2
                if lost_value not in track_set_candidate:
                    track_set_candidate[lost_value] = vehicle_track_set
    if track_set_candidate:
        min_key = min(track_set_candidate)
        [vehicle_track_set.decrement_visit() for vehicle_track_set in vehicle_track_set_list]
        # Reset available visit time
        track_set_candidate[min_key].visit_times = 10
        return track_set_candidate[min_key].track_set
    else:
        return None


def update_track_set(current_image, previous_image, bounding_box):
    """
    Checks if given blob has been detected before. Updated TrackSet if the given blob is not new vehicle; Otherwise,
    append new track set to track_set_list
    """
    track_set = nearest_track_set(bounding_box)
    if track_set:
        track_set = current_image.track(method='mftrack', img=previous_image, ts=track_set, bb=bounding_box)
        print("Track set updated. Now it has {0} blobs.".format(len(track_set)))
        track_set.drawBB(color=(255, 0, 0))
        track_set.drawPath()
        current_image.show()
    else:
        track_set = current_image.track(method='mftrack', img=current_image, bb=bounding_box)
        track_set.drawBB(color=(255, 0, 0))
        track_set.drawPath()
        current_image.show()
        vehicle_track_set_list.append(VehicleTrackSet(track_set))
        print("Found {0} vehicles in total.".format(len(vehicle_track_set_list)))


def main():
    camera = cv2.VideoCapture('video2.avi')
    background_subtractor = cv2.BackgroundSubtractorMOG()

    # Store previous tracking image
    previous_track_image = Image()

    while camera.isOpened():
        is_success, image = camera.read()
        if is_success:
            mask = background_subtractor.apply(image, None, 0.1)
            # Vehicles will be detected from this image
            # track_image = Image(ndimage.median_filter(mask, 3), cv2image=True)
            track_image = Image(mask)
            blobs = track_image.findBlobs(minsize=300, maxsize=800)
            if not blobs:
                # print('No Blobs Found.')
                continue

            # print("Found {0} Blobs. {1}".format(len(blobs)))

            if len(vehicle_track_set_list) == 0:
                # Find first batch of blobs
                for blob in blobs:
                    blob.drawRect(color=Color.BLUE, width=3, alpha=225)
                    # bounding_box = tuple(blob.boundingBox())
                    # print("Area: {0}".format(blob.area()))

                    track_set = track_image.track(method='mftrack', img=track_image, bb=blob.boundingBox())
                    if track_set:
                        vehicle_track_set_list.append(VehicleTrackSet(track_set))
                        track_set.drawBB(color=(255, 0, 0))
                        track_set.drawPath()
                        track_image.show()

            else:
                for blob in blobs:
                    blob.drawRect(color=Color.BLUE, width=3, alpha=225)
                    # print("Blob Coordinate: ({0}, {1}).".format(blob.x, blob.y))
                    update_track_set(track_image, previous_track_image, blob.boundingBox())

            # Save current image
            previous_track_image = track_image

            # time.sleep(0.1)
        else:
            camera.release()
            break


if __name__ == '__main__':
    main()
    print("Finish.")
    cv2.destroyAllWindows()

