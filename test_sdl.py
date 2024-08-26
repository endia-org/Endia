import math
import sdl2
import sdl2.audio
import numpy as np


class AudioGenerator:
    def __init__(
        self,
        frequencies,
        amplitude,
        attack_time,
        release_time,
        lfo_frequency,
        lfo_amplitude,
        lfo_type,
        sample_rate,
    ):
        self.phases = [0] * len(frequencies)
        self.envelope = 0
        self.is_playing = True
        self.lfo_phase = 0
        self.frequencies = frequencies
        self.amplitude = amplitude
        self.attack_time = attack_time
        self.release_time = release_time
        self.lfo_frequency = lfo_frequency
        self.lfo_amplitude = lfo_amplitude
        self.lfo_type = lfo_type
        self.sample_rate = sample_rate

    def generate_samples(self, buffer, samples):
        """
        Generates audio samples and fills the provided NumPy buffer.
        """
        for i in range(samples):
            if self.is_playing:
                if self.envelope < 1.0:
                    self.envelope += 1 / (self.attack_time * self.sample_rate)
                if self.envelope > 1.0:
                    self.envelope = 1.0
            else:
                self.envelope -= 1 / (self.release_time * self.sample_rate)
                if self.envelope < 0.0:
                    self.envelope = 0.0

            # Calculate LFO value
            lfo_value = math.sin(self.lfo_phase) * self.lfo_amplitude
            self.lfo_phase += (
                2 * math.pi * self.lfo_frequency / self.sample_rate
            )
            if self.lfo_phase >= 2 * math.pi:
                self.lfo_phase -= 2 * math.pi

            value = 0
            for j, freq in enumerate(self.frequencies):
                if self.lfo_type == "vibrato":
                    mod_freq = freq * (1 + lfo_value)
                    value += math.sin(self.phases[j]) * self.amplitude
                    self.phases[j] += 2 * math.pi * mod_freq / self.sample_rate
                else:
                    value += math.sin(self.phases[j]) * self.amplitude
                    self.phases[j] += 2 * math.pi * freq / self.sample_rate

                if self.phases[j] >= 2 * math.pi:
                    self.phases[j] -= 2 * math.pi

            if self.lfo_type == "tremolo":
                value *= 1 + lfo_value

            buffer[i] = value * self.envelope


def main():
    # Initialize SDL
    if sdl2.SDL_Init(sdl2.SDL_INIT_AUDIO) != 0:
        print(f"SDL_Init Error: {sdl2.SDL_GetError()}")
        return 1

    # Audio specifications
    SAMPLE_RATE = 44100
    AUDIO_FORMAT = sdl2.audio.AUDIO_F32SYS
    AUDIO_CHANNELS = 1
    AUDIO_SAMPLES = 4096

    # Tone parameters
    FREQUENCIES = [f * 2 ** (-6 / 12) for f in [246.94, 293.66, 369.99, 440]]
    AMPLITUDE = 0.5 / len(FREQUENCIES)

    # Envelope parameters
    ATTACK_TIME = 0.01
    RELEASE_TIME = 0.01

    # LFO parameters
    LFO_FREQUENCY = 4
    LFO_AMPLITUDE = 0.01
    LFO_TYPE = "vibrato"

    audio_generator = AudioGenerator(
        FREQUENCIES,
        AMPLITUDE,
        ATTACK_TIME,
        RELEASE_TIME,
        LFO_FREQUENCY,
        LFO_AMPLITUDE,
        LFO_TYPE,
        SAMPLE_RATE,
    )

    desired = sdl2.audio.SDL_AudioSpec(
        SAMPLE_RATE, AUDIO_FORMAT, AUDIO_CHANNELS, AUDIO_SAMPLES
    )
    obtained = sdl2.audio.SDL_AudioSpec(0, 0, 0, 0)

    dev = sdl2.audio.SDL_OpenAudioDevice(None, 0, desired, obtained, 0)

    if dev == 0:
        print(f"SDL_OpenAudioDevice Error: {sdl2.SDL_GetError()}")
        sdl2.SDL_Quit()
        return 1

    # Allocate NumPy buffer
    buffer = np.zeros(AUDIO_SAMPLES, dtype=np.float32)

    sdl2.audio.SDL_PauseAudioDevice(dev, 0)

    print(f"Playing tones with frequencies: {FREQUENCIES}")
    print(f"LFO: {LFO_TYPE} at {LFO_FREQUENCY} Hz")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            # Check available buffer space
            available = sdl2.audio.SDL_GetQueuedAudioSize(dev)
            if available <= AUDIO_SAMPLES * 4:  # 4 bytes per float32 sample
                # Generate and queue audio data
                audio_generator.generate_samples(buffer, AUDIO_SAMPLES)
                sdl2.audio.SDL_QueueAudio(
                    dev, buffer.tobytes(), len(buffer.tobytes())
                )

            # Your buffer manipulation logic can go here using NumPy operations on the 'buffer'

            sdl2.SDL_Delay(10)

    except KeyboardInterrupt:
        print("\nStopping...")
        audio_generator.is_playing = False
        sdl2.SDL_Delay(int(RELEASE_TIME * 1000))

    sdl2.audio.SDL_CloseAudioDevice(dev)
    sdl2.SDL_Quit()


if __name__ == "__main__":
    main()
