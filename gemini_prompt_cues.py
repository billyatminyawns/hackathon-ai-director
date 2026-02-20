PITCH_CUE = """
Adjusts pitch of the given text. The valid values are integers between -45 and
+100.
The default pitch is 0.
Format:
<pitch value="{value}">{text}</pitch>
Cue tags can be applied on a whole-sentence or word-by-word basis.
"""

TEMPO_CUE = """
Tempo Cue
Adjusts tempo of the given text. The valid values are decimals between 0.7 and
2.3.
The default tempo is 1.
Format:
<tempo value="{value}">{text}</tempo>'
Cue tags can be applied on a whole-sentence or word-by-word basis.
"""


LOUDNESS_CUE = """
Adjusts loudness of the given text. The valid values are integers between -15
and +9.
The default value is a neutral 0.
Format:
<loudness value="{value}">{text}</loudness>
Cue tags can be applied on a whole-sentence or word-by-word basis.
"""


RESPELLING = """
Adjusts the phonetic pronunciation of a word. Used to force a certain
pronunciation, or if the TTS engine is not pronouncing it correctly, most
commonly with things like proper nouns, medical or industry terminology.
Emphasis can be indicated by a phonetic section being in all capital letters.
Each section is delimited by a hyphen, and do not mix upper and lower case
letters in a single phonetic section.

Phonetic respellings reference
Respellings Reference Chart

Vowels
To hear	as in	type	For example,
a	ant	    A	 ::ANT::
a	spa	    AH	 ::SPAH::
a	all	    AW	 ::AWL::
a	eight	AY	::AYT::
e	egg	    EH	::EHG::
e	ease	EE	::EEZ::
i	in	    IH	::IHN::
i	isle	Y	::YL::
o	oat	    OH	::OHT::
o	ooh	    OO	::OO::
o	foot	UU	::FUUT::
u	up	    UH	::UHP::


VOWEL COMBINATIONS
To hear	as in	type	For example,
ar	car	    AR	::KAR::
er	error	ERR	::ERR-ur::
or	more	OR	::MOR::
ow	cow	    OW	::KOW::
oy	oy	    OY	::OY::
ur	urn	    UR	::URN::

Consonants
To hear	as in	type	For example,
b	bunk	B	::BUHNK::
ch	chart	CH	::CHAHRT::
d	dust	D	::DUHST::
f	first	F	::FURST::
g	glow	G	::GLOH::
h	horse	H	::HORS::
j	jell	J	::JEHL::
k	kite	K	::KYT::
l	laugh	K	::LAF::
m	mask	M	::MASK::
n	nest	N	::NEHST::
ng	ring	NG	::RIHNG::
nk	rink	NK	::RIHNK::
p	pop	    P	::PAHP::
qu	quote	KW	::KWOHT::
r	rain	R	::RAYN::
s	slice	S	::SLYS::
sh	shy	    SH	::SHY::
t	tarte	T	::TART::
th	though	DH	::DHOH::
th	think	TH	::THIHNK::
v	van	    V	::VAN::
w	win   	W	::WIHN::
x	axe	    KS	::AKS::
y	yes	    Y	::YEHS::
z	zen  	Z	::ZEHN::
zh	measure	ZH	::MEH-zhur::

Format:
word: original word
phonetic: phonetic respelling to use for the word
<respell value="{phonetic}">{word}</respell>

You cannot repeat phonemes in phonetic repelling using our rules.
"""

POSTPROCESSING = """
I want to feed you an audio file and a design prompt, have the audio file
analyzed and be given back a series of sox ommands to make
the audio sound more like the design prompt.

Do 3 passes of applying effects, analyzing the processed audio and
reapplying effects to make sure we are matching the design aesthetic perfectly.

Lets also add recommendations by default to clean and enhance the audio
to make it sound voice over professional, and to bring the clip loudness
level up to -18 LUFS.

Any audio file that is brought in is converted to 48K hZ with a 32 bit and
make sure that our output is always the same format.

Limit any effects that are not EQ filters, noise reduction, deessing, or
audio clean up to only be applied once and only in the third pass.

Dynamic creative EQ and multi band compression for creative purposes can be
added in the third pass only. Pitch and speed changes can also be added in
phase 3 for design purposes.

In the third pass, support any sound design tooling to bring the outcome to be
as close to the design prompt as possible. 

Please validate that the sox commands with run without raising errors,
consult the documentation if necessary.
"""
