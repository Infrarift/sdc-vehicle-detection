from skimage.feature import hog
from layers.layer import Layer
import cv2

class Hog(Layer):

    def __init__(self, orientations = 6, px_per_cell = 2, cl_per_block = 2):
        self.name = "HOG"
        self.orientations = orientations
        self.px_per_cell = px_per_cell
        self.cl_per_block = cl_per_block
        pass

    def process(self, image_input, image_original, featuremodel, output_path):
        features, hog_image = self.get_hog_features(image_input)
        featuremodel.add_to_current_feature(features)
        return image_input, hog_image

    def get_hog_features(self, image_input):
        in_hog = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
        if self.should_save:
            features, hog_image = hog(in_hog,
                                      orientations=self.orientations,
                                      pixels_per_cell=(self.px_per_cell, self.px_per_cell),
                                      cells_per_block=(self.cl_per_block, self.cl_per_block),
                                      transform_sqrt=False,
                                      visualise=True,
                                      feature_vector=True)
            image_input = cv2.resize(hog_image, (256, 256))

        else:
            features = hog(in_hog,
                           orientations=self.orientations,
                           pixels_per_cell=(self.px_per_cell, self.px_per_cell),
                           cells_per_block=(self.cl_per_block, self.cl_per_block),
                           transform_sqrt=False,
                           visualise=False,
                           feature_vector=True)
        return features, image_input

    def create_hog_subsample(self):
        pass

    def get_hog_subsample(self, featuremodel, hog_vector, window, px_per_cell):
        x_start_window = window[0, 0]
        y_start_window = window[0, 1]
        x_end_window = window[1, 0]
        y_end_window = window[1, 1]

        x_start_hog = x_start_window // px_per_cell
        y_start_hog = y_start_window // px_per_cell
        x_end_hog = x_end_window // px_per_cell
        y_end_hog = y_end_window // px_per_cell

        features = hog_vector[y_start_hog:y_end_hog, x_start_hog:x_end_hog].ravel()
        featuremodel.add_to_current_feature(features)



    def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size,
                  hist_bins):

        draw_img = np.copy(img)
        img = img.astype(np.float32) / 255

        img_tosearch = img[ystart:ystop, :, :]
        ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

        ch1 = ctrans_tosearch[:, :, 0]
        ch2 = ctrans_tosearch[:, :, 1]
        ch3 = ctrans_tosearch[:, :, 2]

        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // pix_per_cell) - 1
        nyblocks = (ch1.shape[0] // pix_per_cell) - 1
        nfeat_per_block = orient * cell_per_block ** 2
        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // pix_per_cell) - 1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step

        # Compute individual channel HOG features for the entire image
        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb * cells_per_step
                xpos = xb * cells_per_step
                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                xleft = xpos * pix_per_cell
                ytop = ypos * pix_per_cell

                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

                # Get color features
                spatial_features = bin_spatial(subimg, size=spatial_size)
                hist_features = color_hist(subimg, nbins=hist_bins)

                # Scale features and make a prediction
                test_features = X_scaler.transform(
                    np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
                # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
                test_prediction = svc.predict(test_features)

                if test_prediction == 1:
                    xbox_left = np.int(xleft * scale)
                    ytop_draw = np.int(ytop * scale)
                    win_draw = np.int(window * scale)
                    cv2.rectangle(draw_img, (xbox_left, ytop_draw + ystart),
                                  (xbox_left + win_draw, ytop_draw + win_draw + ystart), (0, 0, 255), 6)

        return draw_img