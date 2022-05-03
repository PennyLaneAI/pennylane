# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Tests for the QubitUnitary decomposition transforms.
"""

import pytest

import pennylane as qml
from pennylane import numpy as np

from pennylane.wires import Wires

from pennylane.transforms.decompositions import zyz_decomposition
from pennylane.transforms.decompositions import two_qubit_decomposition
from pennylane.transforms.decompositions.two_qubit_unitary import (
    _convert_to_su4,
    _su2su2_to_tensor_products,
    _compute_num_cnots,
)

from test_optimization.utils import check_matrix_equivalence
from gate_data import I, Z, S, T, H, X, CNOT, SWAP

single_qubit_decomps = [
    # First set of gates are diagonal and converted to RZ
    (I, qml.RZ, [0.0]),
    (Z, qml.RZ, [np.pi]),
    (S, qml.RZ, [np.pi / 2]),
    (T, qml.RZ, [np.pi / 4]),
    (qml.RZ(0.3, wires=0).matrix(), qml.RZ, [0.3]),
    (qml.RZ(-0.5, wires=0).matrix(), qml.RZ, [-0.5]),
    # Next set of gates are non-diagonal and decomposed as Rots
    (
        np.array(
            [
                [0, -9.831019270939975e-01 + 0.1830590094588862j],
                [9.831019270939975e-01 + 0.1830590094588862j, 0],
            ]
        ),
        qml.Rot,
        [-0.18409714468526372, np.pi, 0.18409714468526372],
    ),
    (H, qml.Rot, [np.pi, np.pi / 2, 0.0]),
    (X, qml.Rot, [np.pi / 2, np.pi, -np.pi / 2]),
    (qml.Rot(0.2, 0.5, -0.3, wires=0).matrix(), qml.Rot, [0.2, 0.5, -0.3]),
    (
        np.exp(1j * 0.02) * qml.Rot(-1.0, 2.0, -3.0, wires=0).matrix(),
        qml.Rot,
        [-1.0, 2.0, -3.0],
    ),
]


class TestQubitUnitaryZYZDecomposition:
    """Test that the decompositions are correct."""

    @pytest.mark.parametrize("U,expected_gate,expected_params", single_qubit_decomps)
    def test_zyz_decomposition(self, U, expected_gate, expected_params):
        """Test that a one-qubit matrix in isolation is correctly decomposed."""
        obtained_gates = zyz_decomposition(U, Wires("a"))

        assert len(obtained_gates) == 1
        assert isinstance(obtained_gates[0], expected_gate)
        assert obtained_gates[0].wires == Wires("a")

        assert qml.math.allclose(obtained_gates[0].parameters, expected_params, atol=1e-7)

        if obtained_gates[0].num_params == 1:
            obtained_mat = qml.RZ(obtained_gates[0].parameters[0], wires=0).matrix()
        else:
            obtained_mat = qml.Rot(*obtained_gates[0].parameters, wires=0).matrix()

        assert check_matrix_equivalence(obtained_mat, U, atol=1e-7)

    @pytest.mark.torch
    @pytest.mark.parametrize("U,expected_gate,expected_params", single_qubit_decomps)
    def test_zyz_decomposition_torch(self, U, expected_gate, expected_params):
        """Test that a one-qubit operation in Torch is correctly decomposed."""
        import torch

        U = torch.tensor(U, dtype=torch.complex128)

        obtained_gates = zyz_decomposition(U, wire="a")

        assert len(obtained_gates) == 1
        assert isinstance(obtained_gates[0], expected_gate)
        assert obtained_gates[0].wires == Wires("a")

        assert qml.math.allclose(
            qml.math.unwrap(obtained_gates[0].parameters), expected_params, atol=1e-7
        )

        if obtained_gates[0].num_params == 1:
            obtained_mat = qml.RZ(obtained_gates[0].parameters[0], wires=0).matrix()
        else:
            obtained_mat = qml.Rot(*obtained_gates[0].parameters, wires=0).matrix()

        assert check_matrix_equivalence(obtained_mat, qml.math.unwrap(U), atol=1e-7)

    @pytest.mark.tf
    @pytest.mark.parametrize("U,expected_gate,expected_params", single_qubit_decomps)
    def test_zyz_decomposition_tf(self, U, expected_gate, expected_params):
        """Test that a one-qubit operation in Tensorflow is correctly decomposed."""
        import tensorflow as tf

        U = tf.Variable(U, dtype=tf.complex128)

        obtained_gates = zyz_decomposition(U, wire="a")

        assert len(obtained_gates) == 1
        assert isinstance(obtained_gates[0], expected_gate)
        assert obtained_gates[0].wires == Wires("a")

        assert qml.math.allclose(qml.math.unwrap(obtained_gates[0].parameters), expected_params)

        if obtained_gates[0].num_params == 1:
            # With TF and RZ, need to cast since can't just unwrap
            obtained_mat = qml.RZ(obtained_gates[0].parameters[0].numpy(), wires=0).matrix()
        else:
            obtained_mat = qml.Rot(*qml.math.unwrap(obtained_gates[0].parameters), wires=0).matrix()

        assert check_matrix_equivalence(obtained_mat, U, atol=1e-7)

    @pytest.mark.jax
    @pytest.mark.parametrize("U,expected_gate,expected_params", single_qubit_decomps)
    def test_zyz_decomposition_jax(self, U, expected_gate, expected_params):
        """Test that a one-qubit operation in JAX is correctly decomposed."""
        import jax

        # Enable float64 support
        from jax.config import config

        remember = config.read("jax_enable_x64")
        config.update("jax_enable_x64", True)

        U = jax.numpy.array(U, dtype=jax.numpy.complex128)

        obtained_gates = zyz_decomposition(U, wire="a")

        assert len(obtained_gates) == 1
        assert isinstance(obtained_gates[0], expected_gate)
        assert obtained_gates[0].wires == Wires("a")

        assert qml.math.allclose(
            qml.math.unwrap(obtained_gates[0].parameters), expected_params, atol=1e-7
        )

        if obtained_gates[0].num_params == 1:
            obtained_mat = qml.RZ(obtained_gates[0].parameters[0], wires=0).matrix()
        else:
            obtained_mat = qml.Rot(*obtained_gates[0].parameters, wires=0).matrix()

        assert check_matrix_equivalence(obtained_mat, U, atol=1e-7)


# Randomly generated set (scipy.unitary_group) of five U(4) operations.
# These require 3 CNOTs each
samples_3_cnots = [
    # Special case
    SWAP,
    # Unitary from the QMC subroutine
    [
        [0.5, 0.5, 0.5, 0.5],
        [0.5, -0.83333333, 0.16666667, 0.16666667],
        [0.5, 0.16666667, -0.83333333, 0.16666667],
        [0.5, 0.16666667, 0.16666667, -0.83333333],
    ],
    [
        [
            0.5636620105552962 + 0.0965426329099377j,
            -0.5027332564280158 - 0.1635419007447941j,
            -0.3847256335260943 + 0.3721938458489832j,
            -0.0112168300627589 - 0.3268181271182378j,
        ],
        [
            0.212792517319857 - 0.3244649017713778j,
            0.1634380572825265 + 0.0960335485305775j,
            -0.3139573850066476 + 0.2101838768898215j,
            0.5975048356996334 + 0.5601329312675812j,
        ],
        [
            -0.0126840675187197 + 0.0099226785860024j,
            0.400697758142422 - 0.5848686938298225j,
            0.0812309251127118 + 0.561309695149665j,
            -0.3604590752381415 + 0.2133372693485148j,
        ],
        [
            -0.238184683798628 + 0.6822134865739269j,
            -0.0963245167970645 + 0.4155362603090703j,
            0.0955731566481913 + 0.4898062484085562j,
            0.1317657014485092 + 0.1717481576104292j,
        ],
    ],
    [
        [
            0.5135439721958217 + 0.4814385791961366j,
            -0.0469575920616757 - 0.5084648743399148j,
            0.1487574414769012 - 0.2081434671214162j,
            0.4170176952238572 + 0.0662699159748308j,
        ],
        [
            -0.1694469031236633 - 0.5709199999770778j,
            0.1385353297280915 - 0.0224196352047464j,
            0.3755479552341257 - 0.3411293791379575j,
            0.4963367242427539 + 0.3491242747200203j,
        ],
        [
            -0.1340989096401365 - 0.3533610802773053j,
            0.120577125957884 - 0.8395701125658934j,
            -0.0914053397143667 + 0.1816589401138052j,
            -0.3090484419414308 - 0.0295087451853903j,
        ],
        [
            -0.001618511029519 - 0.0835367795138499j,
            0.0053302150713874 + 0.0108880702515039j,
            -0.4304338310290907 + 0.6712032562215353j,
            0.5750401423577074 + 0.1625231252605274j,
        ],
    ],
    [
        [
            -0.0031072463860293 - 0.6094283173470628j,
            -0.1706041375326247 - 0.5095579204727588j,
            0.4717848248508053 - 0.1360136245848086j,
            -0.3088104665532103 + 0.0582056156280258j,
        ],
        [
            0.0649795149056778 - 0.0807006955559929j,
            0.3985010217077551 - 0.4878571486498078j,
            -0.2593500452019464 - 0.369166183684014j,
            0.5611176872453115 + 0.2721364042662399j,
        ],
        [
            0.6706814505431677 - 0.3020432793307177j,
            -0.445130459060895 + 0.2625311675303597j,
            -0.1884345816738757 + 0.0153310963671089j,
            0.1742601869852907 + 0.3546593618463598j,
        ],
        [
            -0.0330058354970538 + 0.2751715747427885j,
            -0.2166842655029602 - 0.0205300200732348j,
            0.6878324043735908 + 0.2156784765639485j,
            0.5870150498641511 + 0.1077050421736986j,
        ],
    ],
    [
        [
            -0.1517007161518054 - 0.3477896338392819j,
            -0.1996932364673036 - 0.3437390801195139j,
            -0.5626622855975293 - 0.1026521198436937j,
            0.5737100114799175 + 0.2042690304628801j,
        ],
        [
            -0.1782045187028614 - 0.1798746573123555j,
            0.6710156796470433 + 0.1482075706151732j,
            -0.4090747335747313 + 0.4717438839366156j,
            -0.2113406078464153 + 0.1706207743654816j,
        ],
        [
            0.0355053593911394 - 0.6453216466524461j,
            -0.1081082592116457 + 0.4107406869820365j,
            0.4773230440109602 + 0.2293807227814043j,
            0.2490211402335333 + 0.2438025872570191j,
        ],
        [
            -0.5071192657577112 - 0.3421211957731553j,
            -0.3794415508044536 - 0.2129960836701763j,
            -0.0100377824272845 - 0.0493528476711767j,
            -0.6523577435362742 + 0.0912843325184198j,
        ],
    ],
    [
        [
            -0.0196408936733692 + 0.0436166575003405j,
            -0.4114171776549808 + 0.1604631422478851j,
            0.8090489042142334 + 0.1854903095749424j,
            -0.0285563913316359 - 0.336031332694153j,
        ],
        [
            0.6111824978145968 - 0.0373559219203255j,
            -0.0607088813252194 - 0.4399072558564469j,
            0.0652009339801155 - 0.3851081946614576j,
            0.4498048829332472 - 0.2701344319737953j,
        ],
        [
            -0.010769501670078 + 0.4210446448669616j,
            -0.1955139271138671 + 0.5982440344653875j,
            -0.3421820362428211 - 0.0107432223507508j,
            0.4746580414982144 - 0.2897923140475613j,
        ],
        [
            0.598418329142766 + 0.2954204464071747j,
            -0.4534575233858022 + 0.0777057134279078j,
            -0.1459545057858948 + 0.1413048185423547j,
            -0.4168424055846296 + 0.3576772276508062j,
        ],
    ],
]

samples_2_cnots = [
    # Special case: CNOT01 CNOT10
    [[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]],
    # CNOT01 HH CNOT01
    [[0.5, 0.5, 0.5, 0.5], [0.5, -0.5, -0.5, 0.5], [0.5, -0.5, 0.5, -0.5], [0.5, 0.5, -0.5, -0.5]],
    # (A \otimes B) CNOT10 CNOT01 (C \otimes D)
    [
        [
            -0.004631217256064 - 0.268534699868393j,
            -0.2794044288596784 - 0.2842241448919913j,
            0.1218071742216182 - 0.7879690440506752j,
            0.2778952482254939 + 0.2367688668871671j,
        ],
        [
            0.4392070528530516 + 0.6904136673657634j,
            -0.1826413830164512 + 0.2570714399459114j,
            -0.1294472995785671 - 0.1887672822622143j,
            -0.0785276120371355 + 0.4152424069619657j,
        ],
        [
            0.1620021717921561 + 0.1884711066839195j,
            0.4872686757195456 - 0.6397614642921917j,
            -0.4411324161480765 - 0.1891338003177969j,
            -0.2059859403259958 - 0.1367802403050884j,
        ],
        [
            0.3022798492102187 - 0.3242755471407363j,
            -0.1551409732131101 + 0.2662930443472138j,
            -0.0292247033180649 - 0.2839975729154084j,
            -0.7363616289920063 - 0.2911260927728274j,
        ],
    ],
    # CNOT01 (A \otimes B) CNOT01 (C \otimes D)
    [
        [
            0.4952546422732242 - 0.3874950570842591j,
            -0.565120684005787 - 0.0812920957189095j,
            0.0599739067738637 - 0.3382684721070105j,
            0.2002937293624679 - 0.3470743738556477j,
        ],
        [
            0.3076748055295139 + 0.1668987026621435j,
            0.1113148363873911 - 0.355244639751341j,
            -0.5552367793191325 + 0.2360956317049145j,
            -0.4402702690966316 - 0.4254695664213973j,
        ],
        [
            -0.0570544619126448 - 0.5997099301388422j,
            0.3690979468442732 - 0.489643527399396j,
            -0.2915922328649661 - 0.1955865908229054j,
            0.1605581200806802 + 0.3347381888568993j,
        ],
        [
            0.3161801218120408 - 0.1384715702727579j,
            0.1624040016253136 + 0.3648031287938079j,
            -0.2674750917123022 + 0.5684412664492574j,
            0.5714177142050261 - 0.0145866404362501j,
        ],
    ],
    # (A \otimes B) CNOT10 (C \otimes D) CNOT01 (E \otimes F)
    [
        [
            -0.5032838832977308 + 0.008462334360567j,
            -0.1994567215589622 + 0.4166138291684532j,
            -0.2927933889967656 + 0.3307978265454508j,
            0.0980847185817039 - 0.5731560630175854j,
        ],
        [
            0.429674866329586 + 0.1517854863780976j,
            -0.260061924207509 - 0.0913938324295813j,
            0.4678474940708733 + 0.4691638242216676j,
            0.4693562225846957 - 0.238881566880469j,
        ],
        [
            -0.3407683276656533 - 0.1781416508017804j,
            0.648130427328909 + 0.1856223456906182j,
            0.3247150132557211 - 0.0284712498269712j,
            0.5276154053566222 + 0.1139531710411058j,
        ],
        [
            -0.6092485765643428 - 0.1411845180727529j,
            -0.3732001418286821 - 0.3418447668630915j,
            0.4811869468078607 + 0.1675038147841858j,
            -0.1994479499053229 + 0.2310173648263156j,
        ],
    ],
]

# These are randomly generated matrices that involve a single CNOT
samples_1_cnot = [
    # Special case
    CNOT,
    # CNOT10 (A \otimes B)
    [
        [
            0.1386390404432302 - 0.334570112999545j,
            0.0148174707054368 - 0.0412250336805805j,
            -0.4992326027350552 + 0.7779384196647746j,
            -0.0558235265127082 + 0.0968780293767465j,
        ],
        [
            0.1110962138948646 + 0.0126194273429225j,
            -0.922378240060485 - 0.0603304126213831j,
            0.0419677169072708 + 0.0125607156508953j,
            -0.3515157103454069 - 0.0871472858321581j,
        ],
        [
            -0.7766806433421076 - 0.5011871453677099j,
            -0.0967373552333893 - 0.0560669491829709j,
            -0.2647978260425042 - 0.2470628571248854j,
            -0.0334214548399949 - 0.0283207908867259j,
        ],
        [
            0.0029378421621844 + 0.0437084651240645j,
            -0.0415219650227516 - 0.3597691905488813j,
            0.0123400367724977 - 0.1111275941460545j,
            -0.0580109080696657 + 0.922527024519649j,
        ],
    ],
    # CNOT01 (A \otimes B)
    [
        [
            0.1275272250362362 - 0.3343853948894475j,
            0.6337142336419559 - 0.1759440058732874j,
            0.2947329435072354 - 0.116226264283067j,
            0.4932993532409792 + 0.309278009290073j,
        ],
        [
            0.3999023544680882 - 0.5221380373177482j,
            -0.3578174135830141 - 0.0065942382104973j,
            0.5800639860980087 - 0.0502285871838717j,
            -0.2092751655259081 - 0.2378654190750314j,
        ],
        [
            -0.1565164704901467 + 0.5608027580207731j,
            0.296176258834552 - 0.1124974497506721j,
            0.5870241263928151 - 0.2965682015252186j,
            -0.3194248225208941 - 0.1613832959085598j,
        ],
        [
            0.0054101937368828 + 0.3167756024402226j,
            -0.4627149049700612 + 0.3534007012683281j,
            0.2601293856027128 - 0.2457834172330788j,
            0.647226333094554 + 0.1168250695361578j,
        ],
    ],
    # (A \otimes B) CNOT01
    [
        [
            0.190458746413919 + 0.2870042726555157j,
            0.502809168004646 + 0.6156507128238845j,
            -0.3763228062935716 + 0.2616231647579805j,
            -0.1734766279722236 + 0.096703042530068j,
        ],
        [
            -0.7237411404174169 - 0.3286968531905625j,
            0.3261596499092991 + 0.1107513854501445j,
            -0.0781992451732389 + 0.1825664178909173j,
            0.2212035729951513 - 0.4014156376735081j,
        ],
        [
            -0.1821207882156272 + 0.0792315428784779j,
            -0.4001569202892715 + 0.2234725352683322j,
            0.3327888523333447 + 0.7218687134048248j,
            0.1125960761054146 + 0.3255274339495012j,
        ],
        [
            0.2594885215726799 - 0.3777978851802577j,
            -0.0957194014420285 + 0.1740213063816186j,
            0.2880779045831403 + 0.1888308965690472j,
            -0.6184873664151125 - 0.4993157690429775j,
        ],
    ],
    # (A \otimes B) CNOT10
    [
        [
            -0.0885673151087696 - 0.0243844138270481j,
            -0.4871418040378778 + 0.4830303555130133j,
            -0.7088634125078255 - 0.1034876301900189j,
            -0.0696323339281462 + 0.0537594674970065j,
        ],
        [
            0.0404303791439687 - 0.0781289108852639j,
            0.3879597260952143 + 0.6022327441469053j,
            0.2377071200057033 - 0.6435221726459008j,
            0.0398405677965372 + 0.082773776650062j,
        ],
        [
            -0.5889549838714252 + 0.4078345916760804j,
            -0.0074407415198821 - 0.0876548780130169j,
            0.0684934101349958 - 0.0612161883111329j,
            -0.0267574702860782 + 0.6854994524959126j,
        ],
        [
            -0.2651157710067235 - 0.6327235490732244j,
            -0.0911552999919856 - 0.011378937851836j,
            0.0437473475396883 + 0.0763211101667442j,
            0.6944805007772098 + 0.1757664963376742j,
        ],
    ],
    # (A \otimes B) CNOT10 (C \otimes D)
    [
        [
            0.2211933286892437 + 0.0021586412960177j,
            0.38444713219383 - 0.0581554847392023j,
            -0.170894684604903 + 0.1031928176859452j,
            -0.1036775219695835 - 0.8656121616196758j,
        ],
        [
            0.3442924845446644 + 0.260151125836039j,
            -0.1914328218155123 - 0.0746552589054714j,
            -0.5744657965166804 - 0.6284808877028109j,
            -0.2036000102332332 + 0.071498266549486j,
        ],
        [
            -0.1048402095394854 + 0.5447532602544797j,
            0.7011096884305793 - 0.024182937133914j,
            -0.1624036601402791 + 0.1839401392571305j,
            -0.1162941782466869 + 0.3554990108217186j,
        ],
        [
            -0.6743897607764879 + 0.0479978633805106j,
            -0.167244245931061 - 0.5352458890411946j,
            -0.4128361178667754 + 0.06711548036308j,
            0.1819390859257292 - 0.1428141352633625j,
        ],
    ],
    # (A \otimes B) CNOT01 (C \otimes D)
    [
        [
            -0.384175417575186 + 0.3923608184977355j,
            0.1002767236561732 - 0.5802482358089508j,
            -0.4052191818361661 + 0.4207032744597814j,
            -0.1025417468296731 + 0.0031874535396264j,
        ],
        [
            -0.4577221128473338 + 0.0586197038286386j,
            -0.0937187062570979 + 0.6742419194459858j,
            -0.0414783182526488 + 0.3776270025012934j,
            -0.2437968467006254 + 0.3462792652304717j,
        ],
        [
            0.3423554335462237 + 0.4267199288134111j,
            0.1796472451791323 + 0.1419455069613175j,
            -0.1490674637388668 - 0.2557411476563918j,
            -0.7409292809972994 - 0.1080757005000481j,
        ],
        [
            0.4279595694962425 - 0.0554066848293549j,
            0.185704890878562 + 0.3208788487100644j,
            -0.4049260830288736 + 0.5127316935400166j,
            0.2406649865130532 - 0.4376672321481798j,
        ],
    ],
]

# Randomly-generated SU(2) x SU(2) matrices. These can be used to test
# the 0-CNOT decomposition case
samples_su2_su2 = [
    # Real-valued case
    (X, H),
    (
        [
            [-0.8951484711495978 + 0.2618482674962029j, 0.1320728980584716 + 0.3357103647583892j],
            [0.1145773085667637 + 0.3420770962361714j, -0.8540795475070558 + 0.3747044530324777j],
        ],
        [
            [0.535259309608487 - 0.7568444138362017j, -0.0856403406729011 - 0.3651708323125105j],
            [0.2346548909477215 + 0.2926108112778731j, -0.9258821807212569 - 0.0453671985080405j],
        ],
    ),
    (
        [
            [-0.3475218186197753 + 0.616967631919303j, -0.6918447778561189 - 0.1411748210564089j],
            [0.4110483325261496 + 0.5741243724791375j, 0.4216047723512865 - 0.5689199321376678j],
        ],
        [
            [-0.1036172252382817 - 0.409539663695392j, 0.0173544935458401 + 0.9062226856837956j],
            [-0.3763539496942646 + 0.8245595424490042j, -0.2579549535044457 + 0.3345422357017644j],
        ],
    ),
    (
        [
            [-0.6199757150883975 - 0.4615217464182165j, -0.5797949307229534 + 0.257809287248686j],
            [0.5072854247342099 - 0.381168319953196j, 0.1016602350822234 + 0.7661836635685451j],
        ],
        [
            [-0.5741575760925136 - 0.4777766752909318j, -0.6560499370893111 + 0.1080324322072939j],
            [0.3761599015364478 - 0.5482483514244275j, 0.1890587151173228 + 0.7226231907949495j],
        ],
    ),
    (
        [
            [0.257810302107603 + 0.3791450460627684j, 0.7491086034534051 - 0.478141383280444j],
            [-0.6878868994174759 + 0.5626673047041733j, -0.2798521896663711 - 0.3631802166496543j],
        ],
        [[np.exp(-1j * np.pi / 3), 0], [0, np.exp(1j * np.pi / 3)]],
    ),
    (
        [
            [0, -1j],
            [-1j, 0],
        ],
        [
            [-0.6429551068755086 - 0.2344852807169338j, 0.348170308379626 + 0.6406268961202303j],
            [-0.0720224467655562 + 0.7255605769552991j, -0.5014071443907067 + 0.4657955472996208j],
        ],
    ),
    (
        [
            [0.7192114465877868 + 0.4233015158437373j, 0.5404227760496959 + 0.1072098172194949j],
            [0.5300962072416202 + 0.1501623549680364j, -0.8330292881813453 + 0.0501147009427765j],
        ],
        [
            [0, -1j],
            [-1j, 0],
        ],
    ),
]


class TestTwoQubitUnitaryDecomposition:
    """Test that two-qubit unitary operations are correctly decomposed."""

    @pytest.mark.parametrize("U", samples_3_cnots)
    def test_convert_to_su4(self, U):
        """Test a matrix in U(4) is correct converted to SU(4)."""
        U_su4 = _convert_to_su4(np.array(U))

        # Ensure the determinant is correct and the mats are equivalent up to a phase
        assert qml.math.isclose(qml.math.linalg.det(U_su4), 1.0)
        assert check_matrix_equivalence(np.array(U), U_su4)

    @pytest.mark.parametrize("U_pair", samples_su2_su2)
    def test_su2su2_to_tensor_products(self, U_pair):
        """Test SU(2) x SU(2) can be correctly factored into tensor products."""
        true_matrix = qml.math.kron(np.array(U_pair[0]), np.array(U_pair[1]))

        A, B = _su2su2_to_tensor_products(true_matrix)

        assert check_matrix_equivalence(qml.math.kron(A, B), true_matrix)

    @pytest.mark.parametrize("wires", [[0, 1], ["a", "b"], [3, 2], ["c", 0]])
    @pytest.mark.parametrize("U", samples_3_cnots)
    def test_two_qubit_decomposition_3_cnots(self, U, wires):
        """Test that a two-qubit matrix using 3 CNOTs is correctly decomposed."""
        U = _convert_to_su4(np.array(U))

        assert _compute_num_cnots(U) == 3

        obtained_decomposition = two_qubit_decomposition(U, wires=wires)
        assert len(obtained_decomposition) == 10

        with qml.tape.QuantumTape() as tape:
            for op in obtained_decomposition:
                qml.apply(op)

        obtained_matrix = qml.matrix(tape, wire_order=wires)

        # We check with a slightly great tolerance threshold here simply because the
        # test matrices were copied in here with reduced precision.
        assert check_matrix_equivalence(U, obtained_matrix, atol=1e-7)

    @pytest.mark.parametrize("wires", [[0, 1], ["a", "b"], [3, 2], ["c", 0]])
    @pytest.mark.parametrize("U", samples_2_cnots)
    def test_two_qubit_decomposition_2_cnots(self, U, wires):
        """Test that a two-qubit matrix using 2 CNOTs isolation is correctly decomposed."""
        U = _convert_to_su4(np.array(U))

        assert _compute_num_cnots(U) == 2

        obtained_decomposition = two_qubit_decomposition(U, wires=wires)
        assert len(obtained_decomposition) == 8

        with qml.tape.QuantumTape() as tape:
            for op in obtained_decomposition:
                qml.apply(op)

        obtained_matrix = qml.matrix(tape, wire_order=wires)

        assert check_matrix_equivalence(U, obtained_matrix, atol=1e-7)

    @pytest.mark.parametrize("wires", [[0, 1], ["a", "b"], [3, 2], ["c", 0]])
    @pytest.mark.parametrize("U", samples_1_cnot)
    def test_two_qubit_decomposition_1_cnot(self, U, wires):
        """Test that a two-qubit matrix using one CNOT is correctly decomposed."""
        U = _convert_to_su4(np.array(U))

        assert _compute_num_cnots(U) == 1

        obtained_decomposition = two_qubit_decomposition(U, wires=wires)
        assert len(obtained_decomposition) == 5

        with qml.tape.QuantumTape() as tape:
            for op in obtained_decomposition:
                qml.apply(op)

        obtained_matrix = qml.matrix(tape, wire_order=wires)

        assert check_matrix_equivalence(U, obtained_matrix, atol=1e-7)

    @pytest.mark.parametrize("wires", [[0, 1], ["a", "b"], [3, 2], ["c", 0]])
    @pytest.mark.parametrize("U_pair", samples_su2_su2)
    def test_two_qubit_decomposition_tensor_products(self, U_pair, wires):
        """Test that a two-qubit tensor product matrix is correctly decomposed."""
        U = _convert_to_su4(qml.math.kron(np.array(U_pair[0]), np.array(U_pair[1])))

        assert _compute_num_cnots(U) == 0

        obtained_decomposition = two_qubit_decomposition(U, wires=wires)
        assert len(obtained_decomposition) == 2

        with qml.tape.QuantumTape() as tape:
            for op in obtained_decomposition:
                qml.apply(op)

        obtained_matrix = qml.matrix(tape, wire_order=wires)

        assert check_matrix_equivalence(U, obtained_matrix, atol=1e-7)


class TestTwoQubitUnitaryDecompositionInterfaces:
    """Test the decomposition in the non-autograd interfaces."""

    @pytest.mark.torch
    @pytest.mark.parametrize("wires", [[0, 1], ["a", "b"], [3, 2], ["c", 0]])
    @pytest.mark.parametrize("U", samples_3_cnots + samples_2_cnots + samples_1_cnot)
    def test_two_qubit_decomposition_torch(self, U, wires):
        """Test that a two-qubit operation in Torch is correctly decomposed."""
        import torch

        U = torch.tensor(U, dtype=torch.complex128)

        obtained_decomposition = two_qubit_decomposition(U, wires=wires)

        with qml.tape.QuantumTape() as tape:
            for op in obtained_decomposition:
                qml.apply(op)

        obtained_matrix = qml.matrix(tape, wire_order=wires)

        assert check_matrix_equivalence(U, obtained_matrix, atol=1e-7)

    @pytest.mark.torch
    @pytest.mark.parametrize("wires", [[0, 1], ["a", "b"], [3, 2], ["c", 0]])
    @pytest.mark.parametrize("U_pair", samples_su2_su2)
    def test_two_qubit_decomposition_tensor_products_torch(self, U_pair, wires):
        """Test that a two-qubit tensor product in Torch is correctly decomposed."""
        import torch

        U1 = torch.tensor(U_pair[0], dtype=torch.complex128)
        U2 = torch.tensor(U_pair[1], dtype=torch.complex128)
        U = qml.math.kron(U1, U2)

        obtained_decomposition = two_qubit_decomposition(U, wires=wires)

        with qml.tape.QuantumTape() as tape:
            for op in obtained_decomposition:
                qml.apply(op)

        obtained_matrix = qml.matrix(tape, wire_order=wires)

        assert check_matrix_equivalence(U, obtained_matrix, atol=1e-7)

    @pytest.mark.tf
    @pytest.mark.parametrize("wires", [[0, 1], ["a", "b"], [3, 2], ["c", 0]])
    @pytest.mark.parametrize("U", samples_3_cnots + samples_2_cnots + samples_1_cnot)
    def test_two_qubit_decomposition_tf(self, U, wires):
        """Test that a two-qubit operation in Tensorflow is correctly decomposed."""
        import tensorflow as tf

        U = tf.Variable(U, dtype=tf.complex128)

        obtained_decomposition = two_qubit_decomposition(U, wires=wires)

        with qml.tape.QuantumTape() as tape:
            for op in obtained_decomposition:
                qml.apply(op)

        obtained_matrix = qml.matrix(tape, wire_order=wires)

        assert check_matrix_equivalence(U, obtained_matrix, atol=1e-7)

    @pytest.mark.tf
    @pytest.mark.parametrize("wires", [[0, 1], ["a", "b"], [3, 2], ["c", 0]])
    @pytest.mark.parametrize("U_pair", samples_su2_su2)
    def test_two_qubit_decomposition_tensor_products_tf(self, U_pair, wires):
        """Test that a two-qubit tensor product in Tensorflow is correctly decomposed."""
        import tensorflow as tf

        U1 = tf.Variable(U_pair[0], dtype=tf.complex128)
        U2 = tf.Variable(U_pair[1], dtype=tf.complex128)
        U = qml.math.kron(U1, U2)

        obtained_decomposition = two_qubit_decomposition(U, wires=wires)

        with qml.tape.QuantumTape() as tape:
            for op in obtained_decomposition:
                qml.apply(op)

        obtained_matrix = qml.matrix(tape, wire_order=wires)

        assert check_matrix_equivalence(U, obtained_matrix, atol=1e-7)

    @pytest.mark.jax
    @pytest.mark.parametrize("wires", [[0, 1], ["a", "b"], [3, 2], ["c", 0]])
    @pytest.mark.parametrize("U", samples_3_cnots + samples_2_cnots + samples_1_cnot)
    def test_two_qubit_decomposition_jax(self, U, wires):
        """Test that a two-qubit operation in JAX is correctly decomposed."""
        import jax
        from jax.config import config

        remember = config.read("jax_enable_x64")
        config.update("jax_enable_x64", True)

        U = jax.numpy.array(U, dtype=jax.numpy.complex128)

        obtained_decomposition = two_qubit_decomposition(U, wires=wires)

        with qml.tape.QuantumTape() as tape:
            for op in obtained_decomposition:
                qml.apply(op)

        obtained_matrix = qml.matrix(tape, wire_order=wires)

        assert check_matrix_equivalence(U, obtained_matrix, atol=1e-7)

    @pytest.mark.jax
    @pytest.mark.parametrize("wires", [[0, 1], ["a", "b"], [3, 2], ["c", 0]])
    @pytest.mark.parametrize("U_pair", samples_su2_su2)
    def test_two_qubit_decomposition_tensor_products_jax(self, U_pair, wires):
        """Test that a two-qubit tensor product in JAX is correctly decomposed."""
        import jax
        from jax.config import config

        remember = config.read("jax_enable_x64")
        config.update("jax_enable_x64", True)

        U1 = jax.numpy.array(U_pair[0], dtype=jax.numpy.complex128)
        U2 = jax.numpy.array(U_pair[1], dtype=jax.numpy.complex128)
        U = qml.math.kron(U1, U2)

        obtained_decomposition = two_qubit_decomposition(U, wires=wires)

        with qml.tape.QuantumTape() as tape:
            for op in obtained_decomposition:
                qml.apply(op)

        obtained_matrix = qml.matrix(tape, wire_order=wires)

        assert check_matrix_equivalence(U, obtained_matrix, atol=1e-7)
