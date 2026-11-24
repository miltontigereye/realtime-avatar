class AuraMicrophone extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffer = new Float32Array(2048);
    this.offset = 0;
    this.enabled = false;
    this.port.onmessage = ({data}) => {
      if (data.flush) {
        if (this.offset) {
          const tail = this.buffer.slice(0,this.offset);
          this.port.postMessage(tail,[tail.buffer]);
        }
        this.offset = 0;this.enabled = false;
        this.port.postMessage({flushed:true});
        return;
      }
      this.enabled = Boolean(data.enabled);
      this.offset = 0;
    };
  }
  process(inputs) {
    const channels = inputs[0];
    if (!this.enabled || !channels?.length) return true;
    for (let i = 0; i < channels[0].length; i++) {
      let value = 0;
      for (const channel of channels) value += channel[i] || 0;
      this.buffer[this.offset++] = value / channels.length;
      if (this.offset === this.buffer.length) {
        this.port.postMessage(this.buffer, [this.buffer.buffer]);
        this.buffer = new Float32Array(2048);
        this.offset = 0;
      }
    }
    return true;
  }
}
registerProcessor('aura-microphone', AuraMicrophone);
