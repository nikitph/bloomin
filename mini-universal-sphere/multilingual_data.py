# ==========================================
# SPANISH DATA (ES)
# ==========================================
SPANISH_AXES = {
    # FUNDAMENTAL
    'existence':    ('mitico', 'real'),  # mythical / real -> Note: using simpler words
    'time':         ('pasado', 'futuro'),
    'valence':      ('malo', 'bueno'),
    'spatial_x':    ('izquierda', 'derecha'),
    'spatial_y':    ('abajo', 'arriba'),
    'spatial_z':    ('lejos', 'cerca'),
    'animacy':      ('objeto', 'animal'), # inanimate/animate approx
    'agency':       ('pasivo', 'activo'),
    
    # PHYSICAL
    'size':         ('pequeno', 'grande'), # normalized chars
    'temperature':  ('frio', 'caliente'),
    'weight':       ('ligero', 'pesado'),
    'hardness':     ('suave', 'duro'),
    'speed':        ('lento', 'rapido'),
    'brightness':   ('oscuro', 'brillante'),
    'strength':     ('debil', 'fuerte'),

    # ABSTRACT
    'concreteness': ('abstracto', 'concreto'),
    'complexity':   ('simple', 'complejo'),
    'certainty':    ('dudoso', 'cierto'), 
    'safety':       ('peligroso', 'seguro'),
    
    # SOCIAL
    'humanness':    ('cosa', 'humano'),
    'gender':       ('mujer', 'hombre'), # feminine/masculine
    'age':          ('viejo', 'joven'),
    'wealth':       ('pobre', 'rico'),
    'politics':     ('conservador', 'liberal'),
}

SPANISH_DATASET = [
    # (Word, Context A, Context B)
    ('banco', 'dinero', 'rio'),       # Bank (money/river)
    ('sirena', 'ambulancia', 'mar'),  # Siren (ambulance/mermaid)
    ('muñeca', 'juguete', 'mano'),    # Doll/Wrist
    ('cabo', 'ejercito', 'geografia'),# Corporal/Cape
    ('cura', 'iglesia', 'medicina'),  # Priest/Cure
    ('carta', 'correo', 'baraja'),    # Letter/Card
    ('cola', 'animal', 'pegamento'),  # Tail/Glue
    ('gato', 'animal', 'coche'),      # Cat/Jack (car tool)
    ('planta', 'jardin', 'pie'),      # Plant/Sole of foot
    ('radio', 'musica', 'hueso'),     # Radio/Radius (bone)
    ('hoja', 'papel', 'arbol'),       # Sheet/Leaf
    ('copa', 'vino', 'arbol'),        # Glass/Tree top
    ('carrera', 'universidad', 'correr'), # Degree/Race
    ('capital', 'ciudad', 'dinero'),  # City/Money
    ('bateria', 'musica', 'coche'),   # Drums/Battery
    ('chile', 'pais', 'comida'),      # Country/Chili
    ('lima', 'ciudad', 'fruta'),      # City/Lime
    ('cierra', 'montana', 'herramienta'), # Sierra(saw)/Mountain (Saw) - Typo fix: Sierra
    ('vino', 'beber', 'venir'),       # Wine/Came (verb) matches different POS
    ('traje', 'ropa', 'traer'),       # Suit/Brought (verb) matches different POS
]
# Fix typos manually in list above during processing if needed, e.g. 'cierra' -> 'sierra'
# Correcting 'cierra' to 'sierra' in the actual list for better results
SPANISH_DATASET[17] = ('sierra', 'montana', 'herramienta')


# ==========================================
# HINDI DATA (HI)
# Note: Requires Transliteration or Devanagari.
# FastText Hindi is usually Devanagari.
# ==========================================
HINDI_AXES = {
    # Using Devanagari script for alignment with cc.hi.300.vec
    'existence':    ('kalpanik', 'vastavik'), # काल्पनिक, वास्तविक (Approx transliteration mapping if model uses it, but usually native)
    # Actually, let's use the Native Script mapping assuming fasttext is native.
    # We will define them in unicode.
    'existence':    ('काल्पनिक', 'वास्तविक'),
    'time':         ('भूत', 'भविष्य'),
    'valence':      ('बुरा', 'अच्छा'),
    'spatial_x':    ('बाएं', 'दाएं'),
    'spatial_y':    ('नीचे', 'ऊपर'),
    'spatial_z':    ('दूर', 'पास'),
    'animacy':      ('वस्तु', 'जीव'),
    'agency':       (' निष्क्रिय', 'सक्रिय'),

    'size':         ('छोटा', 'बड़ा'),
    'temperature':  ('ठंडा', 'गर्म'),
    'weight':       ('हल्का', 'भारी'),
    'speed':        ('धीमा', 'तेज'),
    'strength':     ('कमजोर', 'मजबूत'),

    'concreteness': ('अमूर्त', 'मूर्त'),
    'complexity':   ('सरल', 'जटिल'),
    'safety':       ('खतरनाक', 'सुरक्षित'),

    'humanness':    ('वस्तु', 'इंसान'),
    'gender':       ('स्त्री', 'पुरुष'),
    'age':          ('बूढ़ा', 'जवान'),
    'wealth':       ('गरीब', 'अमीर'),
}

HINDI_DATASET = [
    ('सोना', 'नींद', 'गहना'),       # Sona: Sleep / Gold
    ('उत्तर', 'दिशा', 'प्रश्न'),     # Uttar: North / Answer
    ('कल', 'बीता', 'आने'),          # Kal: Yesterday / Tomorrow (Both time, but distinct)
    ('आम', 'फल', 'साधारण'),         # Aam: Mango / Common
    ('जग', 'पानी', 'दुनिया'),       # Jag: Jug / World
    ('पत्र', 'चिट्ठी', 'पत्ता'),      # Patra: Letter / Leaf
    ('भाग', 'हिंसा', 'दौड़ना'),     # Bhag: Part / Run
    ('मत', 'नहीं', 'वोट'),          # Mat: Don't / Vote
    ('वर', 'दूल्हा', 'वरदान'),      # Var: Groom / Boon
    ('हार', 'गले', 'पराजय'),        # Haar: Necklace / Defeat
    ('तीर', 'बाण', 'किनारा'),       # Teer: Arrow / Bank (river)
    ('पास', 'नजदीक', 'उत्तीर्ण'),    # Paas: Near / Pass (exam)
    ('फल', 'खाने', 'परिणाम'),       # Phal: Fruit / Result
    ('बाल', 'सिर', 'बच्चा'),        # Baal: Hair / Child
    ('लाल', 'रंग', 'beta'),         # Lal: Red / Son (term of endearment)
]
