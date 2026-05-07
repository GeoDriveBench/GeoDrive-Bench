import os
import json

singapore_traffic_handbook = """S1 Driving side & lane discipline: In Singapore you drive on the left. As a default, keep left on two-way roads and dual carriageways unless road markings/signs say otherwise; move right to overtake, to turn right, or to avoid an obstruction, then return left when safe. On roads with two lanes, the left lane is for normal driving and the right lane is mainly for overtaking and right turns; on roads with three lanes, the left lane is for slower vehicles, the centre lane for faster vehicles, and the outer right lane for overtaking and right turns.

S2 Speed limits are in km/h (not mph). Unless otherwise stated, the speed limit on roads in Singapore is generally 50 km/h. Always follow posted limit signs and special reduced-speed zones such as School Zones and Silver Zones.

S3 Traffic lights (vehicles): Standard signals are green, amber, and red. Green means proceed only if the way ahead is clear and the vehicle can fully clear the junction. Drivers must not enter a junction if doing so would cause obstruction (e.g., yellow-box junction). Amber means stop unless too close to stop safely. Red means stop behind the stop line. Even with a green light, drivers must watch for pedestrians, vehicles still clearing the junction, and unexpected hazards.

S4 Left Turn on Red (LTOR): There is no general “turn on red” rule. At junctions with a specific Left Turn on Red sign, drivers may turn left only after making a full stop, giving way to pedestrians, and giving way to traffic approaching from the right before proceeding when safe.

S5 Right-turn control & arrows: Many junctions use dedicated right-turn signals (green arrow). When a red arrow is shown, vehicles must not turn. Right-turning vehicles must give way to oncoming traffic going straight and to pedestrians crossing. Right turns are only permitted when signals or conditions explicitly allow and the path is clear. :contentReference[oaicite:0]{index=0}

S6 Junction priority rules:
- At signalised junctions:
  Follow traffic lights. Vehicles going straight generally have priority over turning vehicles. Turning vehicles must give way to oncoming traffic and pedestrians.
- At unsignalised or uncontrolled junctions:
  Drivers must give way to vehicles approaching from the right when no signals or priority signs are present. :contentReference[oaicite:1]{index=1}
- At major/minor road junctions:
  Vehicles entering from a minor road must give way to traffic on the major road.

S7 Roundabouts: Slow down when approaching a roundabout and give way to traffic already on the roundabout, typically approaching from the right. Enter only when safe and do not block exits.

S8 Yellow-box junctions: Do not enter the yellow box unless the exit road is clear and you can fully clear the junction. This applies even if the traffic light is green.

S9 Pedestrian priority: Drivers must slow down and be prepared to stop for pedestrians at crossings. When turning left or right at a junction, drivers must give way to pedestrians crossing the road into which they are turning.

S10 Bus lanes: Bus lanes operate during specified hours. Other vehicles must not use them during restricted times. Always check roadside signs.

S11 Bus Priority Box / Give Way to Buses: Drivers must give way to buses exiting bus stops where Bus Priority markings are present. Stop before the give-way line and do not block the box.

S12 School Zones & Silver Zones: These are low-speed safety zones. Drivers must reduce speed and watch carefully for vulnerable road users such as children and elderly pedestrians.

S13 Expressways & tunnels: Controlled-access roads where pedestrians are not allowed. Use designated entry/exit ramps and follow lane-use signals. Do not stop except in emergencies.

S14 Lane-use signals & overhead signs: A red “X” indicates a closed lane. Drivers must obey all overhead signals and variable message signs.

S15 Distinctive Singapore road features & signs: Singapore roads commonly include regulatory blue circular signs, GIVE WAY and STOP markings, LTOR signs, right-turn arrows, ERP signs, and bus priority markings. Drivers must always follow specific signs and road markings over general rules.

S16 Driver behavior rules: In Singapore, it is an offence to hold and use a mobile communication device while driving, including while stopped at traffic lights or in a traffic jam; hands-free use is allowed only if the device is not held and you remain in proper control of the vehicle. Riders and pillion riders on motorcycles, including mopeds and scooters, must wear a securely fastened approved protective helmet on all roads. The drink-driving limit for ordinary drivers is 35 micrograms of alcohol per 100 ml of breath or 80 mg per 100 ml of blood (107 mg per 100 ml of urine). Motor vehicles must be fitted with an audible warning device, but the horn may be used only when necessary to warn other road users of danger; do not use it needlessly, and obey any “No sounding horn” signs. Use high beam only when it will not dazzle others; dip your headlights for oncoming traffic, when following another vehicle, and on well-lit roads. Daytime running lights are not generally mandatory for private cars.

S17 Vehicle registration plates: Standard private-car plates in Singapore have black letters and numerals on a white background at the front and black letters and numerals on a yellow background at the rear, normally on reflective plate faces. Ordinary registration numbers use only Latin letters and Arabic numerals; Chinese characters, kanji and mixed-script formats are not used on standard registration plates. For most private cars, the format is an S-series prefix followed by one to four digits and a final checksum letter, for example SBA 1234 X or SMC 12 K. In practice this means an initial S, then one or two serial letters, then the number sequence, with the last letter calculated from a checksum; the prefix is not a regional or city code because Singapore has no regional registration system. Car plates are usually displayed in a single line, but approved two-line layouts are used when the mounting space is too small. There is no EU-style strip, age identifier or regional code panel.

S18 Sign system & visual cues: Singapore uses a UK-style sign system. Most regulatory signs are circular: prohibitions are usually black symbols or numerals on white with a red border (such as speed limits or no-turn signs), while mandatory instructions are commonly white symbols on blue circles; “No Entry” is a red disc with a horizontal white bar. Warning signs are generally red-bordered triangles with a white background and black symbol, not yellow diamond warnings. The STOP sign is the familiar red octagon with a white border and the word “STOP” in English. Give-way control is shown by an inverted white triangle with a red border, often reinforced by “GIVE WAY” road markings. School-area warnings typically use a red-bordered triangular sign showing children, and school zones commonly impose a posted 40 km/h limit, sometimes made more conspicuous with flashing amber beacons and road-surface markings.

S19 Road & pavement markings: In Singapore, lines separating opposing traffic are white, not yellow. A broken white centre line may be crossed when it is safe and legal to do so; a continuous white centre line means you must not cross or straddle it, except for limited needs such as turning into or out of a side road or premises, or passing an obstruction. Where double white centre lines are used, do not overtake or cross the continuous line nearest you; if both are continuous, neither direction may cross. Lane-divider and edge lines are also white: broken white lines separate lanes moving in the same direction, while a solid white edge line marks the carriageway edge, with short broken guide lines at merges and diverges. No-overtaking is therefore shown by solid or double white lines, not by yellow centre markings; yellow kerbside lines in Singapore control parking/waiting. Bus-priority lanes are marked in English, typically BUS LANE or FULL-DAY BUS LANE. Zebra crossings use broad white stripes, often with Belisha beacons on black-and-white poles.

S20 Tolls & special infrastructure: Singapore uses Electronic Road Pricing (ERP): there are no toll booths, and charges are deducted automatically when you pass under an ERP gantry. Vehicles must have the required in-vehicle equipment fitted and a valid payment method available (for example, the appropriate stored-value card or linked account, depending on the IU/OBU system installed). School zones are clearly signed and use a 40 km/h limit when the school-zone controls are active; watch for the amber beacons, school-zone signs and painted carriageway markings, and slow as soon as the lower limit begins. Bus-priority lanes are marked by roadside signs and lane text such as “BUS LANE” or “FULL DAY BUS LANE”. Standard bus lanes operate Monday-Friday 7.30-9.30 am and 5.00-8.00 pm, and Saturday 11.30 am-2.00 pm; full-day bus lanes operate Monday-Saturday 7.30 am-11.00 pm. They are generally not in force on Sundays and public holidays. Do not drive in them during operating hours unless your vehicle is permitted, and give way to buses re-entering from bus bays where give-way markings or signs are provided."""