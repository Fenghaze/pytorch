import pandas as pd



import warnings
warnings.filterwarnings('ignore')

landmarks_frame = pd.read_csv('../data/faces/face_landmarks.csv')

n = 65
img_name = landmarks_frame.iloc[n, 0] # 获取(65, 0)的数据，即第65个标记项的图片名
landmarks = landmarks_frame.iloc[n, 1:].as_matrix()
landmarks = landmarks.astype('float').reshape(-1, 2) # 固定 2 列，多少行不知道

if __name__ == '__main__':
    print('Image name: {}'.format(img_name))
    print('Landmarks shape: {}'.format(landmarks.shape))
    print('First 4 Landmarks: {}'.format(landmarks[:4]))