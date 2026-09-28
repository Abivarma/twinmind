// Converts mic/system audio to 16 kHz mono Int16 PCM in ~100 ms packets.
// The page asks for a 16 kHz AudioContext, so the browser normally resamples for us;
// the linear resampler below only kicks in if the context runs at another rate.
class PcmWorklet extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / 16000;
    this.pos = 1;      // fractional position between prev (0) and current sample (1)
    this.prev = 0;
    this.out = new Int16Array(1600);
    this.n = 0;
    this.sumSq = 0;
  }
  emit(s) {
    s = Math.max(-1, Math.min(1, s));
    this.out[this.n++] = s * 32767;
    this.sumSq += s * s;
    if (this.n === this.out.length) {
      this.port.postMessage({ pcm: this.out.buffer, rms: Math.sqrt(this.sumSq / this.n) }, [this.out.buffer]);
      this.out = new Int16Array(1600);
      this.n = 0;
      this.sumSq = 0;
    }
  }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    for (let k = 0; k < ch.length; k++) {
      const x = ch[k];
      while (this.pos <= 1) {
        this.emit(this.prev + (x - this.prev) * this.pos);
        this.pos += this.ratio;
      }
      this.pos -= 1;
      this.prev = x;
    }
    return true;
  }
}
registerProcessor("pcm-worklet", PcmWorklet);
