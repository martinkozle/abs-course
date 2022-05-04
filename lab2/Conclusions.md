# Conclusions

## Exercise 1

Naj dobri rezultati dade konfiguracijata:

```csv
discount_factor,learning_rate,episodes,iterations,average_steps,average_reward
0.9,0.1,5000,100,12.85,9.3381307238105
```

Generalno site modeli so discount_factor=0.9 i learning_rate=0.1 dadoa podobri rezultati od drugite.

## Exercise 2

Za vtorata zadacha generalnata ideja mi beshe da gi pretvoram state-ovite vo edna dimenzija,
toa go napraviv so formulata y * maxX + x.

No za nitu edna konfiguracija nemam stignato do solution.

## Exercise 3

Za zadacha 3 go pretvorav neprekinatiot domen na state-ot vo edno dimenzionalni diskretni vrednosti koristejki
segmentiranje. Go zimav posledniot position segment koj sto e pomal ili odnakov od neprekinatata vrednost kako `p`.
Na isti nachin go zimav i posledniot speed segment kako `s`. Pa na kraj go pretvoriv vo edno dimenzionalna vrednost
so formulata `p * len(SPEED_SEGMENTS) + s`.

Me interesirashe koj score bi bil prifatliv (dobar), pa najdov GitHub page od nekoj sto go reshil problemot,
i dobil average score od -102.84:

<https://github.com/mshik3/MountainCar-v0>

Po izvrsheni 168 eksperimenti najdobriot score sto jas uspeav da go postignam beshe so beshe -124.64 so konfiguracijata:

```csv
discount_factor,learning_rate,episodes,decay,iterations,average_steps,average_reward
0.9,0.01,10000,0.05,50,124.64,-124.64
```
