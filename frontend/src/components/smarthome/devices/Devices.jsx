import React from 'react';
import Lights from './Lights';
import Fridge from './Fridge';
import TV from './TV';
import AC from './AC';
import Heater from './Heater';
import WashingMachine from './WashingMachine';
import Dishwasher from './Dishwasher';
import Laptop from './Laptop';
import EVCharger from './EVCharger';

export default function Devices({ devicesOn, devicePower, createMat }) {
  return (
    <group>
      <Lights on={!!devicesOn.lights} power={devicePower.lights || 0} />
      <Fridge on={!!devicesOn.fridge} power={devicePower.fridge || 0} createMat={createMat} />
      <TV on={!!devicesOn.tv} power={devicePower.tv || 0} createMat={createMat} />
      <AC on={!!devicesOn.ac} power={devicePower.ac || 0} createMat={createMat} />
      <Heater on={!!devicesOn.heater} power={devicePower.heater || 0} createMat={createMat} />
      <WashingMachine on={!!devicesOn.washing_machine} power={devicePower.washing_machine || 0} createMat={createMat} />
      <Dishwasher on={!!devicesOn.dishwasher} power={devicePower.dishwasher || 0} createMat={createMat} />
      <Laptop on={!!devicesOn.laptop} power={devicePower.laptop || 0} createMat={createMat} />
      <EVCharger on={!!devicesOn.ev_charger} power={devicePower.ev_charger || 0} createMat={createMat} />
    </group>
  );
}

