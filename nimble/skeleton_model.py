__author__ = 'Axel Lefrant'
__email__ = 'axel.lefrant@laposte.net'

skeleton_mapping = \
{
    'HEAD':      0,
    'NECK':      1,
    'RSHOULDER': 2,
    'RELBOW':    3,
    'RWRIST':    4,
    'LSHOULDER': 5,
    'LELBOW':    6,
    'LWRIST':    7,
    'RHIP':      8,
    'RKNEE':     9,
    'RANKLE':    10,
    'LHIP':      11,
    'LKNEE':     12,
    'LANKLE':    13,
    'CHIP':      14
}

n_joints = len(skeleton_mapping)

triplets_angles = \
[
    # Right arm
    ('NECK',      'RSHOULDER', 'RELBOW'),
    ('RSHOULDER', 'RELBOW',    'RWRIST'),
    # Left arm
    ('NECK',      'LSHOULDER', 'LELBOW'),
    ('LSHOULDER', 'LELBOW',    'LWRIST'),
    # Right leg
    ('CHIP',      'RHIP',      'RKNEE'),
    ('RHIP',      'RKNEE',     'RANKLE'),
    # Left leg
    ('CHIP',      'LHIP',      'LKNEE'),
    ('LHIP',      'LKNEE',     'LANKLE'),
    # Horizontal symmetry
    ('HEAD',      'NECK',      'CHIP'),
    # Vertical symmetry
    ('RSHOULDER', 'NECK',      'LSHOULDER'),
    ('RELBOW',    'NECK',      'LELBOW'),
    ('RWRIST',    'NECK',      'LWRIST'),
    ('RKNEE',     'CHIP',      'LKNEE'),
    ('RANKLE',    'CHIP',      'LANKLE'),
    # Hand-Foot
    ('RWRIST',    'CHIP',      'RANKLE'),
    ('LWRIST',    'CHIP',      'LANKLE'),
    ('RWRIST',    'CHIP',      'LANKLE'),
    ('LWRIST',    'CHIP',      'RANKLE')
]

num_triplets = len(triplets_angles)