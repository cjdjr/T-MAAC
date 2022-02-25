from .maddpg import MADDPG
from .sqddpg import SQDDPG
from .iac import IAC
from .iddpg import IDDPG
from .coma import COMA
from .maac import MAAC
from .matd3 import MATD3
from .ippo import IPPO
from .mappo import MAPPO
from .facmaddpg import FACMADDPG
from .safe_maddpg import SafeMADDPG
from .constrained_maddpg import CSMADDPG
from .trans_maddpg import TransMADDPG
from .ind_constrained_maddpg import ICSMADDPG
from .totq_transmaddpg import TotQTransMADDPG
from .encoder_maddpg import ENCMADDPG
from .ics_trans_maddpg import ICSTRANSMADDPG
from .ics_trans_matd3 import ICSTRANSMATD3
from .cs_trans_maddpg import CSTRANSMADDPG
Model = dict(maddpg=MADDPG,
             sqddpg=SQDDPG,
             iac=IAC,
             iddpg=IDDPG,
             coma=COMA,
             maac=MAAC,
             matd3=MATD3,
             ippo=IPPO,
             mappo=MAPPO,
             facmaddpg=FACMADDPG,
             safemaddpg=SafeMADDPG,
             csmaddpg=CSMADDPG,
             transmaddpg=TransMADDPG,
             icsmaddpg=ICSMADDPG,
             totqtransmaddpg=TotQTransMADDPG,
             encmaddpg=ENCMADDPG,
             icstransmaddpg=ICSTRANSMADDPG,
             icstransmatd3=ICSTRANSMATD3,
             cstransmaddpg=CSTRANSMADDPG,
             )

Strategy = dict(maddpg='pg',
                sqddpg='pg',
                iac='pg',
                iddpg='pg',
                coma='pg',
                maac='pg',
                matd3='pg',
                ippo='pg',
                mappo='pg',
                facmaddpg='pg',
                safemaddpg='pg',
                csmaddpg='pg',
                transmaddpg='pg',
                icsmaddpg='pg',
                totqtransmaddpg='pg',
                encmaddpg='pg',
                icstransmaddpg='pg',
                icstransmatd3='pg',
                cstransmaddpg='pg',
                )
