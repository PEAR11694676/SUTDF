from math import pi, e, cos
import torch
from scipy.ndimage import gaussian_filter1d

def add_target_pixel(H):

    a = torch.tensor([0.8114889264106750, 0.7469463348388672, 0.7379295825958252,
         0.8127378821372986, 0.8338796496391296, 0.7468404173851013,
         0.7492808103561401, 0.7781473994255066, 0.7977674603462219,
         0.7728202939033508, 0.7616746425628662, 0.7742629051208496,
         0.7863585948944092, 0.7600508928298950, 0.7509818077087402,
         0.7739070653915405, 0.7672499418258667, 0.7543704509735107,
         0.7496612071990967, 0.7827149629592896, 0.7720717191696167,
         0.7622067928314209, 0.7498849034309387, 0.7821223735809326,
         0.7748042941093445, 0.7599598169326782, 0.7546631693840027,
         0.7835797071456909, 0.7834540009498596, 0.7712513208389282,
         0.7597619891166687, 0.7928839325904846, 0.7955913543701172,
         0.7727344632148743, 0.7697092294692993, 0.8010129332542419,
         0.8079980015754700, 0.7807300090789795, 0.7751091718673706,
         0.7980667948722839, 0.8064836859703064, 0.7773792743682861,
         0.7717233896255493, 0.7980893850326538, 0.8026560544967651,
         0.7750703692436218, 0.7705675363540649, 0.7966925501823425,
         0.8006060719490051, 0.7763776779174805, 0.7721620798110962,
         0.7929226160049438, 0.8025664091110229, 0.7781264185905457,
         0.7732033133506775, 0.7959657311439514, 0.8105497360229492,
         0.7830494046211243, 0.7802311182022095, 0.8017231225967407,
         0.8169331550598145, 0.7944681644439697, 0.7888637185096741,
         0.8100764155387878, 0.8252058029174805, 0.7991917133331299,
         0.7921025156974792, 0.8158374428749084, 0.8304182887077332,
         0.8077691793441772, 0.7944469451904297, 0.8073869347572327,
         0.8324493765830994, 0.8039127588272095, 0.8023118972778320,
         0.8166353106498718, 0.8406638503074646, 0.8119289278984070,
         0.8097251653671265, 0.8192457556724548, 0.8473714590072632,
         0.8257212638854980, 0.8214277029037476, 0.8153425455093384,
         0.8426660895347595, 0.8385217785835266, 0.8429092168807983,
         0.8076218366622925, 0.8654431700706482, 0.8287777900695801,
         0.8701691627502441, 0.7664424180984497, 0.8790050745010376,
         0.8340556621551514, 0.8311355710029602, 0.7206427454948425,
         0.8280810117721558, 0.7450071573257446, 0.7688010334968567,
         0.6897795796394348, 0.7719706296920776, 0.6792286634445190,
         0.6996027231216431, 0.6104327440261841, 0.7438045144081116,
         0.6413324475288391, 0.6822277903556824, 0.6313298344612122,
         0.7595552802085876, 0.6560055613517761, 0.7059485316276550,
         0.6405575275421143, 0.7665123939514160, 0.6661168932914734,
         0.7059628367424011, 0.6362403035163879, 0.7595680952072144,
         0.6681438684463501, 0.6973735094070435, 0.6256343722343445,
         0.7572236061096191, 0.6440950632095337, 0.6964067816734314,
         0.6351513862609863, 0.7681185603141785, 0.6612513661384583,
         0.7133575081825256, 0.6265842914581299, 0.7563774585723877,
         0.6453298330307007, 0.6988064050674438, 0.6121230125427246,
         0.7278666496276855, 0.6054558753967285, 0.6527296900749207,
         0.5669177174568176, 0.6840851902961731, 0.5817876458168030,
         0.6453232169151306, 0.5886268019676208, 0.6853330135345459,
         0.5644986033439636, 0.6259484291076660, 0.5933707356452942,
         0.7474218010902405, 0.6014091968536377, 0.6592584848403931,
         0.5916954874992371, 0.7936887145042419, 0.6815435290336609,
         0.6973249912261963, 0.5936557650566101, 0.8049318790435791,
         0.6541355252265930, 0.6783797144889832, 0.5827887654304504,
         0.8025575280189514, 0.6863169670104980, 0.7709269523620605,
         0.7328007221221924, 0.8623244762420654, 0.7716307640075684,
         0.8049851655960083, 0.7032142877578735, 0.8037629127502441,
         0.6518980860710144, 0.6408451795578003, 0.7258389592170715,
         0.7981150150299072, 0.6287091970443726, 0.7367546558380127,
         0.6784057021141052, 0.7890104055404663, 0.7881496548652649,
         0.5305585265159607, 0.5077259540557861]).to(device='cuda')
    b = torch.tensor([0.2578914165496826, 0.2445842623710632, 0.2443033158779144,
         0.2677795588970184, 0.2836081087589264, 0.2600917816162109,
         0.2578104734420776, 0.2610779106616974, 0.2722148299217224,
         0.2704626917839050, 0.2897734344005585, 0.2939053773880005,
         0.2939680516719818, 0.2867665588855743, 0.2833692133426666,
         0.2977644503116608, 0.2997520864009857, 0.2917241752147675,
         0.2884238660335541, 0.2979332208633423, 0.2938636541366577,
         0.2829162478446960, 0.2814547121524811, 0.2915782630443573,
         0.2888304889202118, 0.2811791002750397, 0.2827196121215820,
         0.2907548844814301, 0.2857272326946259, 0.2743404805660248,
         0.2676606774330139, 0.2762291133403778, 0.2748193144798279,
         0.2629788815975189, 0.2596254348754883, 0.2645854651927948,
         0.2653156220912933, 0.2559118568897247, 0.2557085454463959,
         0.2645995318889618, 0.2673722505569458, 0.2558223307132721,
         0.2557173967361450, 0.2630638778209686, 0.2668578922748566,
         0.2559559047222137, 0.2572357058525085, 0.2648175954818726,
         0.2683146893978119, 0.2583043575286865, 0.2582311630249023,
         0.2662207782268524, 0.2680401504039764, 0.2564911246299744,
         0.2567612826824188, 0.2618217170238495, 0.2625299990177155,
         0.2524780929088593, 0.2504083812236786, 0.2521419525146484,
         0.2570235729217529, 0.2415758371353149, 0.2391878515481949,
         0.2426094561815262, 0.2448560595512390, 0.2341251373291016,
         0.2342259883880615, 0.2339440882205963, 0.2389713078737259,
         0.2312458753585815, 0.2278186976909637, 0.2324534952640533,
         0.2395429462194443, 0.2308992296457291, 0.2272492498159409,
         0.2283757925033569, 0.2343336939811707, 0.2252264618873596,
         0.2218973189592361, 0.2190261036157608, 0.2225576639175415,
         0.2136149108409882, 0.2143747806549072, 0.2089527696371078,
         0.2117048054933548, 0.2131846100091934, 0.2093266695737839,
         0.1944511234760284, 0.2072584778070450, 0.1998649686574936,
         0.1969300806522369, 0.1694594919681549, 0.1863676011562347,
         0.1728033870458603, 0.1675656586885452, 0.1368291527032852,
         0.1507405042648315, 0.1343541592359543, 0.1285339593887329,
         0.1111341193318367, 0.1244653165340424, 0.1027136594057083,
         0.1048304811120033, 0.0883714631199837, 0.0997355431318283,
         0.0839597061276436, 0.0892389267683029, 0.0859434083104134,
         0.1064609661698341, 0.0930068865418434, 0.0969651043415070,
         0.0918858200311661, 0.1116198077797890, 0.0955564305186272,
         0.1005763188004494, 0.0943409129977226, 0.1132189333438873,
         0.0946797430515289, 0.1014563068747520, 0.0893073379993439,
         0.1050437688827515, 0.0894775986671448, 0.0917444601655006,
         0.0875907763838768, 0.1082869842648506, 0.1002561897039413,
         0.1076786071062088, 0.0917042568325996, 0.1066141128540039,
         0.0852456092834473, 0.0891576260328293, 0.0842468217015266,
         0.1017297580838203, 0.0846458077430725, 0.0886475369334221,
         0.0680185779929161, 0.0710419714450836, 0.0549013428390026,
         0.0578089654445648, 0.0527226403355598, 0.0668524280190468,
         0.0556709729135036, 0.0619988404214382, 0.0572222620248795,
         0.0746980458498001, 0.0686964094638824, 0.0790923759341240,
         0.0793104693293571, 0.1027583107352257, 0.1005834564566612,
         0.1206305772066116, 0.0992314070463181, 0.1107854172587395,
         0.0876839384436607, 0.0902134627103806, 0.0704484283924103,
         0.0913626104593277, 0.1160152778029442, 0.1303031295537949,
         0.1279388964176178, 0.1730371117591858, 0.1460751742124557,
         0.1467273980379105, 0.1240277811884880, 0.1333421468734741,
         0.1028142124414444, 0.1067475676536560, 0.0745745152235031,
         0.0973464399576187, 0.0783515572547913, 0.0813313722610474,
         0.0787483751773834, 0.0933093130588531, 0.1054480150341988,
         0.0819473117589951, 0.0927930325269699]).to(device='cuda')
    r_b = torch.tensor([0.014718, 0.015167, 0.015150, 0.015527, 0.015508, 0.015317, 0.015252, 0.015342, 0.015538, 0.015569,
                0.015513, 0.015639, 0.015569, 0.015472, 0.015404, 0.015376, 0.015190, 0.015161, 0.015085, 0.015007,
                0.014999, 0.014950, 0.014951, 0.014801, 0.014876, 0.014866, 0.014770, 0.014825, 0.014811, 0.014820,
                0.014841, 0.014901, 0.014874, 0.014973, 0.014947, 0.015105, 0.015124, 0.015308, 0.015318, 0.015470,
                0.015563, 0.015764, 0.015914, 0.016153, 0.016488, 0.016852, 0.017229, 0.017928, 0.018605, 0.019501,
                0.021006, 0.022542, 0.024429, 0.027729, 0.030904, 0.034651, 0.040668, 0.045924, 0.051871, 0.060028,
                0.065805, 0.072023, 0.081160, 0.087191, 0.092602, 0.099576, 0.104169, 0.108454, 0.113588, 0.117073,
                0.120221, 0.124061, 0.126614, 0.132053, 0.134319, 0.137303, 0.140356, 0.142681, 0.145076, 0.148371,
                0.150959, 0.154537, 0.157417, 0.161388, 0.164491, 0.167750, 0.172250, 0.175710, 0.180516, 0.184192,
                0.189189, 0.193005, 0.196790, 0.201854, 0.205538, 0.210355, 0.214934, 0.218140, 0.222067, 0.224702,
                0.226994, 0.229562, 0.231045, 0.232486, 0.233126, 0.233431, 0.233240, 0.232493, 0.231260, 0.230043,
                0.228169, 0.226528, 0.224176, 0.222321, 0.219703, 0.217707, 0.215042, 0.213026, 0.210374, 0.208430,
                0.205948, 0.204137, 0.201854, 0.200251, 0.198187, 0.196763, 0.194973, 0.193773, 0.192309, 0.191372,
                0.190243, 0.189506, 0.188664, 0.188017, 0.187554, 0.187324, 0.187256, 0.187294, 0.187468, 0.187866,
                0.188299, 0.189010, 0.189918, 0.190987, 0.191914, 0.193345, 0.194543, 0.196278, 0.197711, 0.199733,
                0.201983, 0.204445, 0.206399, 0.209196, 0.212180, 0.214549, 0.217868, 0.221349, 0.224101, 0.227970,
                0.232029, 0.235144, 0.239500, 0.243972, 0.247443, 0.252194, 0.257062, 0.260804, 0.265909, 0.270982,
                0.274832, 0.280087, 0.285247, 0.290366, 0.294283, 0.299747]).to(device='cuda') # iron
    h = H
    # calculate underwater-target reflectance
    u = b / (a + b)
    k = a + b
    r_inf = torch.tensor([0.02719089, 0.0275924, 0.02821107, 0.02877388, 0.02949709, 0.03055052
        , 0.03130687, 0.03205764, 0.03278143, 0.03345723, 0.03415905, 0.03431525
        , 0.03441792, 0.0345802, 0.03448228, 0.03441412, 0.0342991, 0.03414123
        , 0.03398529, 0.03384888, 0.03365626, 0.03348593, 0.03328531, 0.03312509
        , 0.03297441, 0.03281101, 0.03267168, 0.03257622, 0.03251682, 0.03243933
        , 0.03238312, 0.03232143, 0.03228679, 0.03225721, 0.03222481, 0.03218668
        , 0.03215105, 0.03210278, 0.03203293, 0.03199525, 0.03194509, 0.03190374
        , 0.03183933, 0.03177748, 0.03171137, 0.03164548, 0.03161473, 0.03157247
        , 0.03151866, 0.03145717, 0.03140046, 0.0312992, 0.03118534, 0.03102871
        , 0.03082899, 0.03058531, 0.03028168, 0.02993944, 0.02957391, 0.02915798
        , 0.0287298, 0.02832759, 0.02791523, 0.02753372, 0.02718879, 0.02685574
        , 0.02653399, 0.02626685, 0.02602265, 0.02580853, 0.0256194, 0.02542828
        , 0.02526885, 0.02509904, 0.02490972, 0.02471605, 0.02454549, 0.02436036
        , 0.02418822, 0.02401859, 0.02382294, 0.02364252, 0.02343972, 0.02322732
        , 0.02303756, 0.0228516, 0.02262585, 0.02237215, 0.02206462, 0.02169917
        , 0.02128835, 0.02078652, 0.02025354, 0.01966962, 0.01903867, 0.01834252
        , 0.01760906, 0.01690112, 0.01623227, 0.01563544, 0.0151006, 0.01464811
        , 0.01424702, 0.01393585, 0.01368139, 0.01347751, 0.01334772, 0.01324721
        , 0.01316692, 0.0130752, 0.01300631, 0.01294913, 0.01292444, 0.01288206
        , 0.01283927, 0.01283918, 0.01283696, 0.0128407, 0.01283933, 0.01282767
        , 0.01279264, 0.01276876, 0.01271282, 0.01257875, 0.01245887, 0.01229739
        , 0.01212828, 0.01192176, 0.01173327, 0.01155233, 0.01137654, 0.01120074
        , 0.01101604, 0.01093278, 0.01082966, 0.01074002, 0.01064599, 0.01058542
        , 0.01055153, 0.01051294, 0.01045145, 0.01042349, 0.01047051, 0.01050425
        , 0.01056766, 0.01066026, 0.01081694, 0.0110053, 0.01112724, 0.01133854
        , 0.01157059, 0.01177252, 0.0122098, 0.0129539, 0.0136846, 0.0144307
        , 0.01529129, 0.01606132, 0.01697011, 0.01773649, 0.01859136, 0.0193646
        , 0.01979278, 0.01989349, 0.02009806, 0.02046407, 0.02069832, 0.02077156
        , 0.02098315, 0.02140748, 0.02197761, 0.02300836, 0.02339079, 0.02386002
        , 0.02435157, 0.02469808]).to(device='cuda')
    k_uc = 1.03 * (1 + 2.4 * u) ** (0.5) * k
    k_ub = 1.04 * (1 + 5.4 * u) ** (0.5) * k
    r = r_inf * (1 - e ** (-(k + k_uc) * h)) + r_b / pi * e ** (-(k + k_ub) * h)

    return r