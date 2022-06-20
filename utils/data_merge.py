import numpy as np
import sys
sys.path.append('..')
water = np.load('../data/train_data/water.npy')
water_label = np.load('../data/train_data/water_label.npy')

Iron = np.load('../data/train_data/Iron.npy')
Iron_label = np.load('../data/train_data/Iron_label.npy')

Nylon_Carpet = np.load('../data/train_data/Nylon_Carpet.npy')
Nylon_Carpet_label = np.load('../data/train_data/Nylon_Carpet_label.npy')

Plastic_PETE = np.load('../data/train_data/Plastic_PETE.npy')
Plastic_PETE_label = np.load('../data/train_data/Plastic_PETE_label.npy')

Wood_Beam= np.load('../data/train_data/Wood_Beam.npy')
Wood_Beam_label = np.load('../data/train_data/Wood_Beam_label.npy')

#训练样本和标签都是（10000，176）
# print(Iron.shape)
# print(Iron_label.shape)
# print(Nylon_Carpet.shape)
# print(Nylon_Carpet_label.shape)
# print(Plastic_PETE.shape)
# print(Plastic_PETE_label.shape)
# print(Wood_Beam.shape)
# print(Wood_Beam_label.shape)
import matplotlib.pyplot as plt
c= []
c = np.append(Iron,Nylon_Carpet,axis=0)
c = np.append(c,Plastic_PETE,axis=0)
c = np.append(c,Wood_Beam,axis=0)
train_data = np.append(c,water,axis=0)

# print(train_data.shape)  #(38800, 176)
l= []
l = np.append(Iron_label,Nylon_Carpet_label,axis=0)

l = np.append(l,Plastic_PETE_label,axis=0)
l = np.append(l,Wood_Beam_label,axis=0)

label_data = np.append(l,water_label,axis=0)
# print(label_data.shape)  #(38800,)


# print(train_data.shape)
np.save('../data/train_data.npy', train_data)
np.save('..data/label_data.npy', label_data)
# print(train_data[1],l[1])

wavelength = [398.10, 401.30, 404.50, 407.60, 410.80, 414.00, 417.20, 420.40, 423.60, 426.80, 430.00,
              433.30, 436.50, 439.70, 442.90, 446.10, 449.40, 452.60, 455.80, 459.10, 462.30, 465.50,
              468.80, 472.00, 475.30, 478.50, 481.80, 485.10, 488.30, 491.60, 494.90, 498.10, 501.40,
              504.70, 508.00, 511.30, 514.60, 517.80, 521.10, 524.40, 527.70, 531.00, 534.40, 537.70,
              541.00, 544.30, 547.60, 550.90, 554.30, 557.60, 560.90, 564.30, 567.60, 570.90, 574.30,
              577.60, 581.00, 584.30, 587.70, 591.10, 594.40, 597.80, 601.20, 604.50, 607.90, 611.30,
              614.70, 618.10, 621.40, 624.80, 628.20, 631.60, 635.00, 638.40, 641.80, 645.30, 648.70,
              652.10, 655.50, 658.90, 662.40, 665.80, 669.20, 672.70, 676.10, 679.50, 683.00, 686.40,
              689.90, 693.30, 696.80, 700.20, 703.70, 707.20, 710.60, 714.10, 717.60, 721.10, 724.60,
              728.00, 731.50, 735.00, 738.50, 742.00, 745.50, 749.00, 752.50, 756.00, 759.50, 763.10,
              766.60, 770.10, 773.60, 777.10, 780.70, 784.20, 787.80, 791.30, 794.80, 798.40, 801.90,
              805.50, 809.00, 812.60, 816.20, 819.70, 823.30, 826.90, 830.40, 834.00, 837.60, 841.20,
              844.80, 848.40, 852.00, 855.60, 859.20, 862.80, 866.40, 870.00, 873.60, 877.20, 880.80,
              884.40, 888.10, 891.70, 895.30, 899.00, 902.60, 906.20, 909.90, 913.50, 917.20, 920.80,
              924.50, 928.10, 931.80, 935.50, 939.10, 942.80, 946.50, 950.20, 953.80, 957.50, 961.20,
              964.90, 968.60, 972.30, 976.00, 979.70, 983.40, 987.10, 990.80, 994.50, 998.20, 1002.00]
# dict = {0:'water',1:'Iron',2:'Nylon_Carpet',3:"Plastic_PETE",4:"Wood_Beam"}
# t1 = dict[label_data[6500]]
# t2 = dict[label_data[2]]
# t3 = dict[label_data[10002]]
# t4 = dict[label_data[20002]]
# t5 = dict[label_data[30002]]
#
# plt.plot(wavelength, train_data[5000], label=t1, linewidth=2, markersize=5)
# plt.plot(wavelength, train_data[3200], label=t2, linewidth=2,  markersize=5)
# plt.plot(wavelength, train_data[13200], label=t3, linewidth=2,  markersize=5)
# plt.plot(wavelength, train_data[23200], label=t4, linewidth=2,  markersize=5)
# plt.plot(wavelength, train_data[33200], label=t5, linewidth=2,  markersize=5)
# plt.xlabel('band')
# plt.ylabel('reflect value')
# plt.legend()
# plt.title('Target reflectivity')
# plt.show()


