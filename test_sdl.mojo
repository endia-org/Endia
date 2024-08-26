# # import ctypes
# # import math
# # import sdl2
# # import sdl2.audio
# from python import Python, PythonObject
# import math
# import endia as nd
# from memory import memcpy

# struct AudioGenerator:
#     var phase: List[Float32]
#     var envelope: Float32
#     var is_playing: Bool
#     var lfo_phase: Float32
#     var frequencies: List[Float32]
#     var amplitude: Float32
#     var attack_time: Float32
#     var release_time: Float32
#     var lfo_frequency: Float32
#     var lfo_amplitude: Float32
#     var lfo_type: String
#     var sample_rate: Int

#     var AUDIO_SAMPLES: Int
#     var endia_array: nd.Array


#     def __init__(inout self, frequencies: List[Float32], amplitude: Float32, attack_time: Float32, release_time: Float32,
#                  lfo_frequency: Float32, lfo_amplitude: Float32, lfo_type: String, sample_rate: Int, AUDIO_SAMPLES: Int):  # Add sample_rate here

#         self.phase = List[Float32]()
#         for _ in range(len(frequencies)):
#             self.phase.append(0)
#         self.envelope = 0
#         self.is_playing = True
#         self.lfo_phase = 0
#         self.frequencies = frequencies
#         self.amplitude = amplitude
#         self.attack_time = attack_time
#         self.release_time = release_time
#         self.lfo_frequency = lfo_frequency
#         self.lfo_amplitude = lfo_amplitude
#         self.lfo_type = lfo_type
#         self.sample_rate = sample_rate
#         self.AUDIO_SAMPLES = AUDIO_SAMPLES
#         self.endia_array = nd.Array(AUDIO_SAMPLES)

#         print("Phase:", end=" [")
#         for i in range(len(self.phase)):
#             print(self.phase[i], end="")
#             if i < len(self.phase) - 1:
#                 print(", ", end="")
#         print("]")
#         print("Envelope", self.envelope)
#         print("Is playing", self.is_playing)
#         print("LFO Phase", self.lfo_phase)
#         print("Frequencies:", end=" [")
#         for i in range(len(self.frequencies)):
#             print(self.frequencies[i], end="")
#             if i < len(self.frequencies) - 1:
#                 print(", ", end="")
#         print("]")
#         print("Amplitude", self.amplitude)
#         print("Attack time", self.attack_time)
#         print("Release time", self.release_time)
#         print("LFO Frequency", self.lfo_frequency)
#         print("LFO Amplitude", self.lfo_amplitude)
#         print("LFO Type", self.lfo_type)
#         print("Sample rate", self.sample_rate)
#         print("Audio samples", self.AUDIO_SAMPLES)


#     def generate_samples(inout self, np_buffer: PythonObject, samples: PythonObject):
#         """
#         Generates audio samples and fills the provided buffer.
#         """

#         memcpy_numpy_to_endia(np_buffer, self.endia_array)

#         buffer = self.endia_array.data()

#         for i in range(int(samples)):
#             if self.is_playing:
#                 if self.envelope < 1.0:
#                     self.envelope += 1 / (self.attack_time * self.sample_rate)  # Use self.sample_rate
#                 if self.envelope > 1.0:
#                     self.envelope = 1.0
#             else:
#                 self.envelope -= 1 / (self.release_time * self.sample_rate)  # Use self.sample_rate
#                 if self.envelope < 0.0:
#                     self.envelope = 0.0

#             # Calculate LFO value
#             var pi = Float32(3.14159265358979323846)
#             lfo_value = math.sin(self.lfo_phase) * self.lfo_amplitude
#             self.lfo_phase += 2 * pi * self.lfo_frequency / self.sample_rate  # Use self.sample_rate
#             if self.lfo_phase >= 2 * pi:
#                 self.lfo_phase -= 2 * pi

#             value = Float32(0)
#             for j in range(len(self.frequencies)):
#                 if self.lfo_type == "vibrato":
#                     mod_freq = self.frequencies[j] * (1 + lfo_value)
#                     value += math.sin(self.phase[j]) * self.amplitude
#                     self.phase[j] += 2 * pi * mod_freq / self.sample_rate  # Use self.sample_rate
#                 else:
#                     value += math.sin(self.phase[j]) * self.amplitude
#                     self.phase[j] += 2 * pi * self.frequencies[j] / self.sample_rate  # Use self.sample_rate

#                 if self.phase[j] >= 2 * pi:
#                     self.phase[j] -= 2 * pi

#             if self.lfo_type == "tremolo":
#                 value *= (1 + lfo_value)

#             buffer[i] = (value * self.envelope).cast[DType.float32]()

#         memcpy_endia_to_numpy(self.endia_array, np_buffer)


# @always_inline
# fn get_np_dtype[dtype: DType](np: PythonObject) raises -> PythonObject:
#     @parameter
#     if dtype.__is__(DType.float32):
#         return np.float32
#     elif dtype.__is__(DType.float64):
#         return np.int32
#     elif dtype.__is__(DType.int32):
#         return np.int64
#     elif dtype.__is__(DType.int64):
#         return np.uint8

#     raise "Unknown datatype"


# @always_inline
# fn memcpy_endia_to_numpy(src: nd.Array, dst: PythonObject) raises:
#     var dst_ptr = dst.__array_interface__["data"][0].unsafe_get_as_pointer[
#         DType.float32
#     ]()
#     var src_data = src.data()
#     var length = src.size()
#     memcpy(dst_ptr, src_data, length)

# @always_inline
# fn memcpy_numpy_to_endia(src: PythonObject, dst: nd.Array) raises:
#     var src_ptr = src.__array_interface__["data"][0].unsafe_get_as_pointer[
#         DType.float32
#     ]()
#     var dst_ptr = dst.data()
#     var length = dst.size()
#     memcpy(dst_ptr, src_ptr, length)


# def main():
#     sdl2 = Python.import_module("sdl2")

#     # Initialize SDL
#     sdl2.SDL_Init(sdl2.SDL_INIT_AUDIO)

#     # Audio specifications
#     SAMPLE_RATE = 2**15
#     AUDIO_FORMAT = sdl2.audio.AUDIO_F32SYS
#     AUDIO_CHANNELS = 1
#     AUDIO_SAMPLES = 2**10

#     # Tone parameters
#     FREQUENCIES = List[Float32](246.94, 293.66, 369.99, 440)
#     semitone_shift = -6
#     for i in range(len(FREQUENCIES)):
#         FREQUENCIES[i] = FREQUENCIES[i] * Float32(2) ** (semitone_shift/ Float32(12))
#     AMPLITUDE = Float32(0.5) / len(FREQUENCIES)

#     # Envelope parameters
#     ATTACK_TIME = Float32(0.01)
#     RELEASE_TIME = Float32(0.01)

#     # LFO parameters
#     LFO_FREQUENCY = Float32(4)
#     LFO_AMPLITUDE = Float32(0.01)
#     LFO_TYPE = String("vibrato")

#     audio_generator = AudioGenerator(FREQUENCIES, AMPLITUDE, ATTACK_TIME, RELEASE_TIME,
#                                     LFO_FREQUENCY, LFO_AMPLITUDE, LFO_TYPE, SAMPLE_RATE, AUDIO_SAMPLES)

#     desired = sdl2.audio.SDL_AudioSpec(
#         SAMPLE_RATE, AUDIO_FORMAT, AUDIO_CHANNELS, AUDIO_SAMPLES
#     )
#     obtained = sdl2.audio.SDL_AudioSpec(0, 0, 0, 0)

#     dev = sdl2.audio.SDL_OpenAudioDevice(None, 0, desired, obtained, 0)

#     if dev == 0:
#         # print(f"SDL_OpenAudioDevice Error: {sdl2.SDL_GetError()}")
#         sdl2.SDL_Quit()
#         return 1

#     # Allocate buffer
#     np = Python.import_module("numpy")
#     buffer = np.zeros(AUDIO_SAMPLES, dtype=np.float32)

#     sdl2.audio.SDL_PauseAudioDevice(dev, 0)  # Start audio playback

#     print("LFO:", LFO_TYPE, "at", LFO_FREQUENCY, "Hz")
#     print("Press Ctrl+C to stop.")

#     try:
#         while True:
#             # Check available buffer space
#             available = sdl2.audio.SDL_GetQueuedAudioSize(dev)
#             if available <= AUDIO_SAMPLES * 4:  # 4 bytes per float32 sample
#                 # Generate and queue audio data
#                 audio_generator.generate_samples(buffer, AUDIO_SAMPLES)
#                 sdl2.audio.SDL_QueueAudio(dev, buffer.tobytes(), len(buffer.tobytes()))

#             sdl2.SDL_Delay(1)  # Adjust delay as needed

#     except KeyboardInterrupt:
#         print("\nStopping...")
#         audio_generator.is_playing = False
#         sdl2.SDL_Delay(int(RELEASE_TIME * 1000))

#     sdl2.audio.SDL_CloseAudioDevice(dev)
#     sdl2.SDL_Quit()


from python import Python, PythonObject
import math
import endia as nd
from memory import memcpy


struct AudioGenerator:
    var phase: List[Float32]
    var envelope: Float32
    var is_playing: Bool
    var lfo_phase: Float32
    var frequencies: List[Float32]
    var amplitudes: List[Float32]
    var attack_time: Float32
    var release_time: Float32
    var lfo_frequency: Float32
    var lfo_amplitude: Float32
    var lfo_type: String
    var sample_rate: Int
    var AUDIO_SAMPLES: Int
    var endia_array: nd.Array
    var MAX_AMPLITUDE: Float32

    def __init__(
        inout self,
        frequencies: List[Float32],
        amplitudes: List[Float32],
        attack_time: Float32,
        release_time: Float32,
        lfo_frequency: Float32,
        lfo_amplitude: Float32,
        lfo_type: String,
        sample_rate: Int,
        AUDIO_SAMPLES: Int,
        MAX_AMPLITUDE: Float32,
    ):
        self.phase = List[Float32]()
        for _ in range(len(frequencies)):
            self.phase.append(0)
        self.envelope = 0
        self.is_playing = True
        self.lfo_phase = 0
        self.frequencies = frequencies
        self.amplitudes = amplitudes
        self.attack_time = attack_time
        self.release_time = release_time
        self.lfo_frequency = lfo_frequency
        self.lfo_amplitude = lfo_amplitude
        self.lfo_type = lfo_type
        self.sample_rate = sample_rate
        self.AUDIO_SAMPLES = AUDIO_SAMPLES
        self.endia_array = nd.Array(AUDIO_SAMPLES)
        self.MAX_AMPLITUDE = MAX_AMPLITUDE

        # Print initialization details...
        print("Phase:", end=" [")
        for i in range(len(self.phase)):
            print(self.phase[i], end="")
            if i < len(self.phase) - 1:
                print(", ", end="")
        print("]")
        print("Envelope", self.envelope)
        print("Is playing", self.is_playing)
        print("LFO Phase", self.lfo_phase)
        print("Frequencies:", end=" [")
        for i in range(len(self.frequencies)):
            print(self.frequencies[i], end="")
            if i < len(self.frequencies) - 1:
                print(", ", end="")
        print("]")
        print("Amplitudes:", end=" [")
        for i in range(len(self.amplitudes)):
            print(self.amplitudes[i], end="")
            if i < len(self.amplitudes) - 1:
                print(", ", end="")
        print("]")
        print("Attack time", self.attack_time)
        print("Release time", self.release_time)
        print("LFO Frequency", self.lfo_frequency)
        print("LFO Amplitude", self.lfo_amplitude)
        print("LFO Type", self.lfo_type)
        print("Sample rate", self.sample_rate)
        print("Audio samples", self.AUDIO_SAMPLES)

    def generate_samples(
        inout self, np_buffer: PythonObject, samples: PythonObject
    ):
        memcpy_numpy_to_endia(np_buffer, self.endia_array)
        buffer = self.endia_array.data()

        var pi = Float32(3.14159265358979323846)

        for i in range(int(samples)):
            if self.is_playing:
                if self.envelope < 1.0:
                    self.envelope += 1 / (self.attack_time * self.sample_rate)
                if self.envelope > 1.0:
                    self.envelope = 1.0
            else:
                self.envelope -= 1 / (self.release_time * self.sample_rate)
                if self.envelope < 0.0:
                    self.envelope = 0.0

            lfo_value = math.sin(self.lfo_phase) * self.lfo_amplitude
            self.lfo_phase += 2 * pi * self.lfo_frequency / self.sample_rate
            if self.lfo_phase >= 2 * pi:
                self.lfo_phase -= 2 * pi

            value = Float32(0)
            for j in range(len(self.frequencies)):
                if self.lfo_type == "vibrato":
                    mod_freq = self.frequencies[j] * (1 + lfo_value)
                    value += math.sin(self.phase[j]) * self.amplitudes[j]
                    self.phase[j] += 2 * pi * mod_freq / self.sample_rate
                else:
                    value += math.sin(self.phase[j]) * self.amplitudes[j]
                    self.phase[j] += (
                        2 * pi * self.frequencies[j] / self.sample_rate
                    )

                if self.phase[j] >= 2 * pi:
                    self.phase[j] -= 2 * pi

            if self.lfo_type == "tremolo":
                value *= 1 + lfo_value

            # Normalize the value to MAX_AMPLITUDE
            if abs(value) > self.MAX_AMPLITUDE:
                value = (value / abs(value)) * self.MAX_AMPLITUDE

            buffer[i] = (value * self.envelope).cast[DType.float32]()

        memcpy_endia_to_numpy(self.endia_array, np_buffer)


@always_inline
fn get_np_dtype[dtype: DType](np: PythonObject) raises -> PythonObject:
    @parameter
    if dtype.__is__(DType.float32):
        return np.float32
    elif dtype.__is__(DType.float64):
        return np.int32
    elif dtype.__is__(DType.int32):
        return np.int64
    elif dtype.__is__(DType.int64):
        return np.uint8

    raise "Unknown datatype"


@always_inline
fn memcpy_endia_to_numpy(src: nd.Array, dst: PythonObject) raises:
    var dst_ptr = dst.__array_interface__["data"][0].unsafe_get_as_pointer[
        DType.float32
    ]()
    var src_data = src.data()
    var length = src.size()
    memcpy(dst_ptr, src_data, length)


@always_inline
fn memcpy_numpy_to_endia(src: PythonObject, dst: nd.Array) raises:
    var src_ptr = src.__array_interface__["data"][0].unsafe_get_as_pointer[
        DType.float32
    ]()
    var dst_ptr = dst.data()
    var length = dst.size()
    memcpy(dst_ptr, src_ptr, length)


def main():
    sdl2 = Python.import_module("sdl2")

    # Initialize SDL
    sdl2.SDL_Init(sdl2.SDL_INIT_AUDIO)

    # Audio specifications
    SAMPLE_RATE = 2**13
    AUDIO_FORMAT = sdl2.audio.AUDIO_F32SYS
    AUDIO_CHANNELS = 1
    AUDIO_SAMPLES = 2**10

    # Fundamental frequency (E1 - deep bass)
    fundamental = Float32(33.20) * 2

    # Extended harmonic series with slight inharmonicity for richness
    FREQUENCIES = List[Float32](
        fundamental,
        fundamental * 1.1,
        fundamental * 1.2,
        fundamental * 1.3,
        fundamental * 1.04,
        fundamental * 1.05,
        fundamental * 1.06,
        fundamental * 1.07,
        fundamental * 1.08,
        fundamental * 1.09,
        fundamental * 1.01,
        fundamental * 1.011,
    )

    # Amplitudes for a rich, complex bass sound
    AMPLITUDES = List[Float32](
        0.6,  # Fundamental
        0.4,  # 2nd harmonic (octave)
        0.25,  # 3rd harmonic
        0.18,  # 4th harmonic
        0.15,  # 5th harmonic
        0.12,  # 6th harmonic
        0.1,  # 7th harmonic
        0.08,  # 8th harmonic
        0.06,  # 9th harmonic
        0.04,  # 10th harmonic
        0.02,  # 12th harmonic
        0.01,  # 14th harmonic
    )

    # optionally: transpose the frequencies
    semitone_shift = Float32(0)
    if semitone_shift != 0:
        for i in range(len(FREQUENCIES)):
            FREQUENCIES[i] = FREQUENCIES[i] * Float32(2) ** (
                semitone_shift / Float32(12)
            )

    # Maximum overall amplitude
    MAX_AMPLITUDE = Float32(0.9)

    # Envelope parameters
    ATTACK_TIME = Float32(0.01)
    RELEASE_TIME = Float32(0.01)

    # LFO parameters
    LFO_FREQUENCY = Float32(0)
    LFO_AMPLITUDE = Float32(0.0)
    LFO_TYPE = String("vibrato")

    audio_generator = AudioGenerator(
        FREQUENCIES,
        AMPLITUDES,
        ATTACK_TIME,
        RELEASE_TIME,
        LFO_FREQUENCY,
        LFO_AMPLITUDE,
        LFO_TYPE,
        SAMPLE_RATE,
        AUDIO_SAMPLES,
        MAX_AMPLITUDE,
    )

    desired = sdl2.audio.SDL_AudioSpec(
        SAMPLE_RATE, AUDIO_FORMAT, AUDIO_CHANNELS, AUDIO_SAMPLES
    )
    obtained = sdl2.audio.SDL_AudioSpec(0, 0, 0, 0)

    dev = sdl2.audio.SDL_OpenAudioDevice(None, 0, desired, obtained, 0)

    if dev == 0:
        sdl2.SDL_Quit()
        return 1

    # Allocate buffer
    np = Python.import_module("numpy")
    buffer = np.zeros(AUDIO_SAMPLES, dtype=np.float32)

    sdl2.audio.SDL_PauseAudioDevice(dev, 0)  # Start audio playback

    print("LFO:", LFO_TYPE, "at", LFO_FREQUENCY, "Hz")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            available = sdl2.audio.SDL_GetQueuedAudioSize(dev)
            if available <= AUDIO_SAMPLES * 4:
                audio_generator.generate_samples(buffer, AUDIO_SAMPLES)
                sdl2.audio.SDL_QueueAudio(
                    dev, buffer.tobytes(), len(buffer.tobytes())
                )

            sdl2.SDL_Delay(1)

    except KeyboardInterrupt:
        print("\nStopping...")
        audio_generator.is_playing = False
        sdl2.SDL_Delay(int(RELEASE_TIME * 1000))

    sdl2.audio.SDL_CloseAudioDevice(dev)
    sdl2.SDL_Quit()
