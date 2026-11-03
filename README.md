# xPersona

**A sentence in, a realistic 3D body out (.glb).** A [HotMotor](https://github.com/Joosehot/HotMotor) engine: a closed vocabulary, a parser, one rule per decision, a beam search with judges, and checks before anything is written. The shape itself comes from a network trained once on MakeHuman's bodies and then kept frozen.

```
$ xpersona "a tall slim woman with long arms and a round head" -o out/woman.glb
$ xpersona "a woman with a very beautiful face"
$ xpersona "a short stocky man with thick legs" --explain
```

## How it works

1. **Words → decisions.** The rules decide height, build, proportions, torso shape, head, face and pose, every number from `rules.toml`. Each step says which words asked for it.
2. **Decisions → measures.** The measures a body of that sex, muscle and weight is expected to have, changed by what was asked, and held to what MakeHuman's bodies of that build actually measure.
3. **Measures → shape.** A network (designed and trained with Neuras) gives the body's 96 main shape components from sex, muscle, weight, nine measures, the head shape, the proportions value and 85 face values. The input is aimed: the made body is measured and the input moved by what it missed.
4. **Checks.** Closed and facing out, one piece, no loops, height, on the ground, symmetric, the legs and arm span within human ranges, every measure as asked, and the face measured against the average face.

"Beautiful" means two stated things, not a verdict: MakeHuman's ideal-proportions canon for the body, and for the face averageness and symmetry, the traits research ties most firmly to attractive faces (Langlois & Roggman 1990; Rhodes 2006).

## Training

```
xpersona --train 8000      # about 40 s; writes model/persona.bin
```

8000 young-adult bodies are composed from MakeHuman's targets with random values and measured; one in ten is held out. On those unseen bodies the mean vertex error is 11.0 mm (the face 3.5 mm).

The training data is not in this repository: it is MakeHuman's base mesh and targets (CC0 1.0, [makehumancommunity/makehuman](https://github.com/makehumancommunity/makehuman)), fetched into `data/makehuman` with a sparse clone of `makehuman/data/3dobjs`, `targets` and `modifiers`. The trained model in `model/` is derived from that CC0 data.

## Viewing

```
python -m http.server 8000
# open http://localhost:8000/viewer/?file=../out/woman.glb
```

## Building

xPersona depends on Neuras by path (`../Neuras`), which is not published yet, so for now it builds only next to a Neuras checkout.

## Limits

- Young adults only; no hair, eyebrows, clothes or skin colour; one material (clay) plus the eyes.
- The template's A-pose only: a T-pose needs a rig.
- `--sculpt` makes the older body from blended shapes by the rules alone, without the network.

## License

MIT, see [LICENSE](LICENSE). © 2026 Joose Hotari. MakeHuman's assets are CC0 1.0.
