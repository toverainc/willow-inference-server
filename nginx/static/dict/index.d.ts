declare module 'dictation_support/dictation_device' {
  /**
   * @license
   * Copyright 2022 Google LLC
   *
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   */
  import { FootControlDevice } from 'dictation_support/foot_control_device';
  import { PowerMic3Device } from 'dictation_support/powermic_3_device';
  import { SpeechMikeGamepadDevice } from 'dictation_support/speechmike_gamepad_device';
  import { SpeechMikeHidDevice } from 'dictation_support/speechmike_hid_device';
  export type DictationDevice = SpeechMikeHidDevice | SpeechMikeGamepadDevice | PowerMic3Device | FootControlDevice;

}
declare module 'dictation_support/dictation_device_base' {
  /**
   * @license
   * Copyright 2022 Google LLC
   *
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   */
  /// <reference types="w3c-web-hid" />
  import { DictationDevice } from 'dictation_support/dictation_device';
  export enum ImplementationType {
      SPEECHMIKE_HID = 0,
      SPEECHMIKE_GAMEPAD = 1,
      FOOT_CONTROL = 2,
      POWERMIC_3 = 3
  }
  export enum DeviceType {
      UNKNOWN = 0,
      FOOT_CONTROL_ACC_2310_2320 = 6212,
      FOOT_CONTROL_ACC_2330 = 2330,
      SPEECHMIKE_LFH_3200 = 3200,
      SPEECHMIKE_LFH_3210 = 3210,
      SPEECHMIKE_LFH_3220 = 3220,
      SPEECHMIKE_LFH_3300 = 3300,
      SPEECHMIKE_LFH_3310 = 3310,
      SPEECHMIKE_LFH_3500 = 3500,
      SPEECHMIKE_LFH_3510 = 3510,
      SPEECHMIKE_LFH_3520 = 3520,
      SPEECHMIKE_LFH_3600 = 3600,
      SPEECHMIKE_LFH_3610 = 3610,
      SPEECHMIKE_SMP_3700 = 3700,
      SPEECHMIKE_SMP_3710 = 3710,
      SPEECHMIKE_SMP_3720 = 3720,
      SPEECHMIKE_SMP_3800 = 3800,
      SPEECHMIKE_SMP_3810 = 3810,
      SPEECHMIKE_SMP_4000 = 4000,
      SPEECHMIKE_SMP_4010 = 4010,
      SPEECHONE_PSM_6000 = 6001,
      POWERMIC_3 = 4097,
      POWERMIC_4 = 100
  }
  export enum ButtonEvent {
      NONE = 0,
      REWIND = 1,
      PLAY = 2,
      FORWARD = 4,
      INS_OVR = 16,
      RECORD = 32,
      COMMAND = 64,
      STOP = 256,
      INSTR = 512,
      F1_A = 1024,
      F2_B = 2048,
      F3_C = 4096,
      F4_D = 8192,
      EOL_PRIO = 16384,
      TRANSCRIBE = 32768,
      TAB_BACKWARD = 65536,
      TAB_FORWARD = 131072,
      CUSTOM_LEFT = 262144,
      CUSTOM_RIGHT = 524288,
      ENTER_SELECT = 1048576,
      SCAN_END = 2097152,
      SCAN_SUCCESS = 4194304
  }
  export type ButtonEventListener = (device: DictationDevice, bitMask: ButtonEvent) => void | Promise<void>;
  export abstract class DictationDeviceBase {
      readonly hidDevice: HIDDevice;
      private static next_id;
      readonly id: number;
      abstract readonly implType: ImplementationType;
      protected readonly buttonEventListeners: Set<ButtonEventListener>;
      protected lastBitMask: number;
      protected readonly onInputReportHandler: (event: HIDInputReportEvent) => Promise<void>;
      protected constructor(hidDevice: HIDDevice);
      init(): Promise<void>;
      shutdown(closeDevice?: boolean): Promise<void>;
      addButtonEventListener(listener: ButtonEventListener): void;
      protected onInputReport(event: HIDInputReportEvent): Promise<void>;
      protected handleButtonPress(data: DataView): Promise<void>;
      protected filterOutputBitMask(outputBitMask: number): number;
      abstract getDeviceType(): DeviceType;
      protected abstract getButtonMappings(): Map<ButtonEvent, number>;
      protected abstract getInputBitmask(data: DataView): number;
      protected abstract getThisAsDictationDevice(): DictationDevice;
  }

}
declare module 'dictation_support/dictation_device_manager' {
  /**
   * @license
   * Copyright 2022 Google LLC
   *
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   */
  /// <reference types="w3c-web-hid" />
  import { DictationDevice } from 'dictation_support/dictation_device';
  import { ButtonEventListener } from 'dictation_support/dictation_device_base';
  import { SpeechMikeGamepadDevice } from 'dictation_support/speechmike_gamepad_device';
  import { MotionEventListener } from 'dictation_support/speechmike_hid_device';
  type DeviceEventListener = (device: DictationDevice) => void | Promise<void>;
  export class DictationDeviceManager {
      protected readonly hidApi: HID;
      protected readonly buttonEventListeners: Set<ButtonEventListener>;
      protected readonly deviceConnectEventListeners: Set<DeviceEventListener>;
      protected readonly deviceDisconnectEventListeners: Set<DeviceEventListener>;
      protected readonly motionEventListeners: Set<MotionEventListener>;
      protected readonly devices: Map<HIDDevice, DictationDevice>;
      protected readonly pendingProxyDevices: Map<HIDDevice, SpeechMikeGamepadDevice>;
      protected readonly onConnectHandler: (event: HIDConnectionEvent) => Promise<void>;
      protected readonly onDisconectHandler: (event: HIDConnectionEvent) => Promise<void>;
      protected isInitialized: boolean;
      constructor(hidApi?: HID);
      getDevices(): DictationDevice[];
      init(): Promise<void>;
      shutdown(): Promise<void>;
      requestDevice(): Promise<Array<DictationDevice>>;
      addButtonEventListener(listener: ButtonEventListener): void;
      addDeviceConnectedEventListener(listener: DeviceEventListener): void;
      addDeviceDisconnectedEventListener(listener: DeviceEventListener): void;
      addMotionEventListener(listener: MotionEventListener): void;
      protected failIfNotInitialized(): void;
      protected createAndAddInitializedDevices(hidDevices: HIDDevice[]): Promise<Array<DictationDevice>>;
      protected createDevice(hidDevice: HIDDevice): Promise<DictationDevice | undefined>;
      protected assignPendingProxyDevices(): void;
      protected addListeners(device: DictationDevice): void;
      protected onHidDeviceConnected(event: HIDConnectionEvent): Promise<void>;
      protected onHidDeviceDisconnected(event: HIDConnectionEvent): Promise<void>;
  }
  export {};

}
declare module 'dictation_support/foot_control_device' {
  /**
   * @license
   * Copyright 2022 Google LLC
   *
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   */
  /// <reference types="w3c-web-hid" />
  import { ButtonEvent, DeviceType, DictationDeviceBase, ImplementationType } from 'dictation_support/dictation_device_base';
  export class FootControlDevice extends DictationDeviceBase {
      readonly implType = ImplementationType.FOOT_CONTROL;
      static create(hidDevice: HIDDevice): FootControlDevice;
      getDeviceType(): DeviceType;
      protected getButtonMappings(): Map<ButtonEvent, number>;
      protected getInputBitmask(data: DataView): number;
      protected getThisAsDictationDevice(): FootControlDevice;
  }

}
declare module 'dictation_support/index' {
  /**
   * @license
   * Copyright 2022 Google LLC
   *
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   */
  import { ButtonEvent as ButtonEvent_, DeviceType as DeviceType_, ImplementationType as ImplementationType_ } from 'dictation_support/dictation_device_base';
  import { DictationDeviceManager as DictationDeviceManager_ } from 'dictation_support/dictation_device_manager';
  import { FootControlDevice as FootControlDevice_ } from 'dictation_support/foot_control_device';
  import { LedStatePM3 as LedStatePM3_, PowerMic3Device as PowerMic3Device_ } from 'dictation_support/powermic_3_device';
  import { SpeechMikeGamepadDevice as SpeechMikeGamepadDevice_ } from 'dictation_support/speechmike_gamepad_device';
  import { EventMode as EventMode_, LedIndex as LedIndex_, LedMode as LedMode_, MotionEvent as MotionEvent_, SimpleLedState as SimpleLedState_ } from 'dictation_support/speechmike_hid_device';
  export namespace DictationSupport {
      const ImplementationType: typeof ImplementationType_;
      const DeviceType: typeof DeviceType_;
      const ButtonEvent: typeof ButtonEvent_;
      const DictationDeviceManager: typeof DictationDeviceManager_;
      const FootControlDevice: typeof FootControlDevice_;
      const LedStatePM3: typeof LedStatePM3_;
      const PowerMic3Device: typeof PowerMic3Device_;
      const SpeechMikeGamepadDevice: typeof SpeechMikeGamepadDevice_;
      const EventMode: typeof EventMode_;
      const SimpleLedState: typeof SimpleLedState_;
      const LedIndex: typeof LedIndex_;
      const LedMode: typeof LedMode_;
      const MotionEvent: typeof MotionEvent_;
  }

}
declare module 'dictation_support/powermic_3_device' {
  /**
   * @license
   * Copyright 2022 Google LLC
   *
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   */
  /// <reference types="w3c-web-hid" />
  import { ButtonEvent, DeviceType, DictationDeviceBase, ImplementationType } from 'dictation_support/dictation_device_base';
  export enum LedStatePM3 {
      OFF = 0,
      RED = 1,
      GREEN = 2
  }
  export class PowerMic3Device extends DictationDeviceBase {
      readonly implType = ImplementationType.POWERMIC_3;
      static create(hidDevice: HIDDevice): PowerMic3Device;
      getDeviceType(): DeviceType;
      setLed(state: LedStatePM3): Promise<void>;
      protected getButtonMappings(): Map<ButtonEvent, number>;
      protected getInputBitmask(data: DataView): number;
      protected getThisAsDictationDevice(): PowerMic3Device;
  }

}
declare module 'dictation_support/speechmike_gamepad_device' {
  /**
   * @license
   * Copyright 2022 Google LLC
   *
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   */
  /// <reference types="w3c-web-hid" />
  import { ButtonEvent, DeviceType, DictationDeviceBase, ImplementationType } from 'dictation_support/dictation_device_base';
  export class SpeechMikeGamepadDevice extends DictationDeviceBase {
      readonly implType = ImplementationType.SPEECHMIKE_GAMEPAD;
      static create(hidDevice: HIDDevice): SpeechMikeGamepadDevice;
      getDeviceType(): DeviceType;
      protected getButtonMappings(): Map<ButtonEvent, number>;
      protected getInputBitmask(data: DataView): number;
      protected getThisAsDictationDevice(): SpeechMikeGamepadDevice;
  }

}
declare module 'dictation_support/speechmike_hid_device' {
  /**
   * @license
   * Copyright 2022 Google LLC
   *
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      http://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   */
  /// <reference types="w3c-web-hid" />
  import { DictationDevice } from 'dictation_support/dictation_device';
  import { ButtonEvent, DeviceType, DictationDeviceBase, ImplementationType } from 'dictation_support/dictation_device_base';
  import { SpeechMikeGamepadDevice } from 'dictation_support/speechmike_gamepad_device';
  export enum EventMode {
      HID = 0,
      KEYBOARD = 1,
      BROWSER = 2,
      WINDOWS_SR = 3,
      DRAGON_FOR_MAC = 4,
      DRAGON_FOR_WINDOWS = 5
  }
  export enum SimpleLedState {
      OFF = 0,
      RECORD_INSERT = 1,
      RECORD_OVERWRITE = 2,
      RECORD_STANDBY_INSERT = 3,
      RECORD_STANDBY_OVERWRITE = 4
  }
  export enum LedIndex {
      RECORD_LED_GREEN = 0,
      RECORD_LED_RED = 1,
      INSTRUCTION_LED_GREEN = 2,
      INSTRUCTION_LED_RED = 3,
      INS_OWR_BUTTON_LED_GREEN = 4,
      INS_OWR_BUTTON_LED_RED = 5,
      F1_BUTTON_LED = 6,
      F2_BUTTON_LED = 7,
      F3_BUTTON_LED = 8,
      F4_BUTTON_LED = 9
  }
  export enum LedMode {
      OFF = 0,
      BLINK_SLOW = 1,
      BLINK_FAST = 2,
      ON = 3
  }
  export type LedState = Record<LedIndex, LedMode>;
  export enum MotionEvent {
      PICKED_UP = 0,
      LAYED_DOWN = 1
  }
  export type MotionEventListener = (device: DictationDevice, event: MotionEvent) => void | Promise<void>;
  enum Command {
      SET_LED = 2,
      SET_EVENT_MODE = 13,
      BUTTON_PRESS_EVENT = 128,
      IS_SPEECHMIKE_PREMIUM = 131,
      GET_DEVICE_CODE_SM3 = 135,
      GET_DEVICE_CODE_SMP = 139,
      GET_DEVICE_CODE_SO = 150,
      GET_EVENT_MODE = 141,
      WIRELESS_STATUS_EVENT = 148,
      MOTION_EVENT = 158
  }
  export class SpeechMikeHidDevice extends DictationDeviceBase {
      readonly implType = ImplementationType.SPEECHMIKE_HID;
      protected deviceCode: number;
      protected ledState: LedState;
      protected sliderBitsFilter: number;
      protected lastSliderValue: number;
      protected lastButtonValue: number;
      protected commandResolvers: Map<Command, (data: DataView) => void>;
      protected commandTimeouts: Map<Command, number>;
      protected readonly motionEventListeners: Set<MotionEventListener>;
      protected proxyDevice: SpeechMikeGamepadDevice | undefined;
      static create(hidDevice: HIDDevice): SpeechMikeHidDevice;
      init(): Promise<void>;
      shutdown(): Promise<void>;
      addMotionEventListener(listener: MotionEventListener): void;
      getDeviceCode(): number;
      getDeviceType(): DeviceType;
      setSimpleLedState(simpleLedState: SimpleLedState): Promise<void>;
      setLed(index: LedIndex, mode: LedMode): Promise<void>;
      protected sendLedState(): Promise<void>;
      assignProxyDevice(proxyDevice: SpeechMikeGamepadDevice): void;
      protected onProxyButtonEvent(bitMask: ButtonEvent): Promise<void>;
      protected handleCommandResponse(command: Command, data: DataView): Promise<void>;
      getEventMode(): Promise<EventMode>;
      setEventMode(eventMode: EventMode): Promise<void>;
      protected fetchDeviceCode(): Promise<void>;
      protected determineSliderBitsFilter(): void;
      protected onInputReport(event: HIDInputReportEvent): Promise<void>;
      protected getButtonMappings(): Map<ButtonEvent, number>;
      protected getInputBitmask(data: DataView): number;
      protected getThisAsDictationDevice(): SpeechMikeHidDevice;
      protected handleMotionEvent(data: DataView): Promise<void>;
      protected sendCommand(command: Command, input?: number[]): Promise<void>;
      protected sendCommandAndWaitForResponse(command: Command, input?: number[]): Promise<DataView>;
      protected filterOutputBitMask(outputBitMask: number): number;
  }
  export {};

}
declare module 'dictation_support' {
  import main = require('dictation_support/index');
  export = main;
}