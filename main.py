import numpy, numba, math, cv2
from matplotlib import pyplot


def init():
    circle = numpy.array([
        [0,0,0,0,0,1,1,1,1,1,0,0,0,0,0],
        [0,0,0,1,1,1,1,1,1,1,1,1,0,0,0],
        [0,0,1,1,1,1,1,1,1,1,1,1,1,0,0],
        [0,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
        [0,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [0,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
        [0,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
        [0,0,1,1,1,1,1,1,1,1,1,1,1,0,0],
        [0,0,0,1,1,1,1,1,1,1,1,1,0,0,0],
        [0,0,0,0,0,1,1,1,1,1,0,0,0,0,0]
    ], dtype='uint8')

    def get_grain_shape_custom(fname, d, dots):
        grain = cv2.imread(fname)
        assert grain.shape[0] == grain.shape[1], 'Изображение должно быть квадратным!'
        grain = (cv2.resize(grain, (int(d * dots), int(d * dots))).sum(axis=2) < 383).astype('uint8')
        area = numpy.zeros(grain.shape, dtype='uint8')
        cv2.circle(area, (area.shape[1] // 2, area.shape[0] // 2), area.shape[1] // 2, 1, -1)
        area = area.astype(bool)
        grain[~area] = 1
        return grain, area

    def get_grain_shape(d, hole, l, ignited_faces, dots):
        f = numpy.ndarray((int((d - hole) * .5 * dots), int(l * dots)), dtype='uint8')
        f[...] = 1
        assert ignited_faces in (0, 1, 2), 'ignited_faces должно быть 0, 1 или 2'
        assert hole > 0. or 0 < ignited_faces, 'Должно быть или hole > 0, или ignited_faces > 0'
        if ignited_faces == 1:
            f[:, 0] = 2
        elif ignited_faces == 2:
            f[:, 0] = f[:, f.shape[1] - 1] = 2
        if hole > 0.:
            f[0, :] = 2
        return f

    def step_custom(grain, l, area, dots, rad, density, h): # масса сгорающего за временной шаг топлива, г
        grain2 = grain.copy()
        cv2.erode(grain2, circle, grain)
        n = int((grain[area] != grain2[area]).sum())
        return (n / dots ** 2 * l + rad * int(grain[area].sum()) / dots ** 3 * 2) * 1e-3 * density * h

    @numba.jit
    def step(f, f1, dots, hole, density, h): # масса сгорающего за временной шаг топлива, г
        n = 0.
        for i in range(f.shape[0]):
            for j in range(f.shape[1]):
                if f1[i, j] == 2:
                    for x, y in [(i, j-1), (i+1, j-1), (i+1, j), (i+1, j+1), (i, j+1), (i-1, j+1), (i-1, j), (i-1, j-1)]:
                        if 0 <= x < f.shape[0] and 0 <= y < f.shape[1]:
                            if f[x, y] == 1:
                                f[x, y] = 2
                    n += (i * 2. + hole * dots)
                    f[i, j] = 0
        return n * math.pi * .001 * dots ** (-3) * density * h

    @numba.jit
    def T2(t1, p1, p2, k): # температура на выходе
        return t1 * p2 ** (1. - 1. / k) * p1 ** (1. / k - 1.)

    @numba.jit
    def U2(t1, t2, mol, k, R=8.314): # скорость потока на выходе
        u22 = R / mol * 2 * k / (k - 1) * (t1 - t2)
        return math.sqrt(u22)

    @numba.jit
    def S1_s2(p1, p2, k): # критика / выход сопла
        a = ((k + 1.) * .5) ** (2./(k-1.)) * ((k + 1.) / (k - 1.)) / p1 ** (2./k)
        b = p2 ** (2./k) - p2 ** (1./k+1.) / p1 ** (1.-1./k)
        return (a * b) ** .5

    @numba.jit
    def S1_s2_d(p1, p2, k): # производная S1_s2 по p2
        a = ((k + 1.) * .5) ** (2./(k-1.)) * ((k + 1.) / (k - 1.)) / p1 ** (2./k)
        b = 2. / k * p2 ** (2./k-1.) - (1. + 1. / k) * p2 ** (1./k) / p1 ** (1.-1./k)
        return a * b / S1_s2(p1, p2, k) * .5

    @numba.jit
    def P2(p1, s1, s2, k): # давление на выходе
        p2 = 1e-40
        x = s1 / s2
        for _ in range(10):
            dx = S1_s2_d(p1, p2, k)
            p2 += (x - S1_s2(p1, p2, k)) / dx
        return p2

    @numba.jit
    def M(mol, p2, t2, s2, u2, R=8.314): # масса выхлопа, выходящего за секунду
        return (mol * p2 / t2 / R) * s2 * u2

    @numba.jit
    def P_pe_u_s2_true_t2(m, mol, s1, s2, t1, k, dynamic_nozzle_out, nozzle_friction, out_pressure, R=8.314): # давление и скорость выходящих газов
        p1 = 1.; s2_true = s2
        for _ in range(2):
            p2 = P2(p1, s1, s2_true, k)
            t2 = T2(t1, p1, p2, k)
            u2 = U2(t1, t2, mol, k, R=R) * nozzle_friction
            m2 = M(mol, p2, t2, s2_true, u2, R=R)
            p1 *= m / m2
            if dynamic_nozzle_out: s2_true = min(s2, s1 / S1_s2(p1, 101325. * out_pressure, k))
        return p1, p2, u2, s2_true, t2

    def show(x, y, title, xlabel, ylabel, x2=None, y2=None):
            fig, ax = pyplot.subplots()
            ax.axhline(y=0, color='k'); ax.axvline(x=0, color='k')
            fig.suptitle(title); pyplot.xlabel(xlabel); pyplot.ylabel(ylabel)
            pyplot.plot(x, y)
            if not x2 is None:
                pyplot.plot(x2, y2, linestyle='dotted')
            pyplot.show()

    def main(custom_grain, mol, smoke, I, smoke_c, dots, density, density_t, d, d_t, h, hole, l, l_t, ignited_faces, min_pressure, use_min_pressure, nozzle_throat, d_throat, nozzle_out, d_out, speed_k, friction, a, a_t, n, n_t, dynamic_nozzle_out, out_pressure, nozzle_friction, thrust_loss, timestep, t, mass, k, **_kwargs):
        if custom_grain is ...: rad = 1
        else: rad = 7

        if not type(k) in (int, float):
            gas_cv = I * .5 * 8.314 / (mol * .001 * (1. - smoke))
            gas_y = (I + 2) / I
            gas_cp = gas_y * gas_cv
            k = (gas_cp + smoke_c) / (gas_cv + smoke_c)
            print('показатель адиабаты:', k)

        if custom_grain is ...:
            grain = get_grain_shape(d, hole, l, ignited_faces, dots)
            x = step(grain, numpy.copy(grain), dots, hole, density, h)
        else:
            grain, area = get_grain_shape_custom(custom_grain, d, dots)
            img = numpy.ndarray(grain.shape + (3,), dtype='uint8')
            img[...] = 255
            ya1, xa1 = numpy.where(grain)
            img[ya1, xa1] = (((xa1 + ya1) % (dots * 2) > dots).astype('uint8') * 32 + 64).reshape(xa1.shape + (1,)).repeat(3, axis=1)
            img[cv2.erode(grain, circle) != grain] = 255, 0, 0
            img[~area] = 160
            img[cv2.dilate(area.astype('uint8'), circle) != area] = 0
            pyplot.imshow(img)
            pyplot.title('Форма канала в топливной шашке')
            pyplot.show()
            l2 = l
            x = step_custom(grain, l2, area, dots, rad, density, h)
            #area_sum = area.sum()
        i = 0
        Y = []
        while x > 0.:
            Y.append(x)
            i += 1
            if custom_grain is ...:
                x = step(grain, numpy.copy(grain), dots, hole, density, h)
            else:
                #if grain[area].sum() / area_sum <= min_propellant:
                #    break
                l2 -= rad / dots * 2.
                x = step_custom(grain, l2, area, dots, rad, density, h)

        propellant = sum(Y)
        print('масса топлива:', propellant, 'г')

        w = True
        s1 = nozzle_throat ** 2 * 0.25 * math.pi * 1e-6
        s2 = nozzle_out    ** 2 * 0.25 * math.pi * 1e-6
        ds1 = (nozzle_throat + d_throat) ** 2 * 0.25 * math.pi * 1e-6 - s1
        ds2 = (nozzle_out    + d_out   ) ** 2 * 0.25 * math.pi * 1e-6 - s2
        X = [0.]; Yp = [0.]; Yt = [0.]; Yt2 = []; Ym = [0.]; Ykn = [0.]; Ykn_t = [0.]
        i1 = 0.; time = 0.
        l_t_2 = l_t; time_t = ...
        while i1 < i and (Yp[-1] >= min_pressure or i1 == 0. or custom_grain is ... or not use_min_pressure):
            r = 1.; r_t = 1.; p = 1.
            for _ in range(20):
                m = 0. # масса сгорающего топлива, г/с
                if i1 < len(Y): m += Y[int(i1)] * dots / rad * r * speed_k
                if l_t_2 > 0.: m += d_t ** 2 * .25e-3 * math.pi * r_t * density_t
                p, pe, u, s2_true, t2 = P_pe_u_s2_true_t2(m * .001, mol * .001, s1, s2, t, k, dynamic_nozzle_out, nozzle_friction, out_pressure)
                r = a * (p * 1e-6) ** n; r_t = a_t * (p * 1e-6) ** n_t
            X.append(time)
            Yp.append(p * 9.87e-6 - 1.)
            #print(pe * 9.87e-6, u, m, t2)
            Yt.append((u * m * .001 + s2_true * (pe - 101325.)) / 9.8 * thrust_loss)
            Yt2.append(t2)
            Ym.append(Ym[-1] - m * timestep)
            Ykn.append(Y[int(i1)] / density * dots / rad * 1e-3 / s1)
            if l_t_2 > 0.: Ykn_t.append((Y[int(i1)] / density * dots / rad * 1e-3 + d_t ** 2 * .25e-6 * math.pi) / s1)
            else: Ykn_t.append(Ykn[-1])
            i1 += r * dots / rad * timestep * speed_k
            if l_t_2 > 0.: l_t_2 -= r_t * timestep
            elif time_t is ...: time_t = time
            time += timestep
            s1 += ds1 * m / propellant * timestep; s2 += ds2 * m / propellant * timestep
            if (not type(Yt[-1]) is float) and w:
                print('\033[31m!!!что-то не так. Проверьте параметры!!!\033[0m')
                w = False
            elif Yt[-1] < 0. and w:
                print('\033[31m!!!что-то не так. Тяга отрицательная. Проверьте параметры!!!\033[0m')
                w = False
        X.append(time); Yp.append(0.); Yt.append(0.); Ym.append(Ym[-1]); Ykn.append(0.); Ykn_t.append(0.)
        Ym = [m - Ym[-1] for m in Ym]

        Yh = []; Xh = []
        rocket_speed = 0.; height = 0.; max_speed = 0; max_g = 0.
        i = 0
        while rocket_speed >= 0. or i < len(Yt) or height >= 0.:
            Yh.append(height)
            Xh.append(i * timestep)
            height += rocket_speed * timestep
            ax = 0.
            if i < len(Yt):
                mass_current = mass - Ym[0] + Ym[i]
                ax += Yt[i] * 9.8e+3 / mass_current
            ax -= rocket_speed ** 2 * ((rocket_speed > 0.) * 2. - 1.) * friction / mass_current * 1000. + 9.8
            rocket_speed += ax * timestep
            max_speed = max(rocket_speed, max_speed)
            max_g = max(ax / 9.8 + 1., max_g)
            i += 1

        if time_t is ...: time_t = X[-1] + l_t_2 / (a_t * 101325e-6 ** n_t)
        print('время работы движка:'                                          , X[-1]                                                     , 'с')
        if l_t > 0: print('время горения трассера:'                           , time_t                                                    , 'с')
        print('максимальная тяга:'                                            , max(Yt)                                                   , 'кгс')
        print('средняя тяга:'                                                 , sum(Yt) / len(Yt)                                         , 'кгс')
        print('импульс:'                                                      , sum(Yt) * 9.8 * timestep                                  , 'кг*м/с')
        print('удельный импульс:'                                             , sum(Yt) * 9.8 * timestep / propellant * 1000.             , 'м*с')
        print('!давление относительно атмосферного, а не абсолютное!')
        print('максимальное давление:'                                        , max(Yp)                                                   , 'атм')
        print('среднее давление:'                                             , sum(Yp) / len(Yp)                                         , 'атм')
        print('оптимальный диаметр выхода сопла (для среднего давления):'     , nozzle_throat / S1_s2(sum(Yp) / len(Yp) + 1., 1., k) ** .5, 'мм')
        print('оптимальный диаметр выхода сопла (для максимального давления):', nozzle_throat / S1_s2(max(Yp)           + 1., 1., k) ** .5, 'мм')
        print('Kn - отношение площади горения топлива к площади критики сопла')
        print('максимальный Kn' + ['', ' (без трассера)'][l_t > 0.] + ':'     , max(Ykn))
        print('средний Kn' + ['', ' (без трассера)'][l_t > 0.] + ':'          , sum(Ykn) / len(Ykn))
        if l_t > 0.:
            print('максимальный Kn (с трассером):'                            , max(Ykn_t))
            print('средний Kn (с трассером):'                                 , sum(Ykn_t) / len(Ykn_t))
        print('высота полёта:'                                                , max(Yh)                                                   , 'м')
        if max_speed > 335.:
            print('\033[31m!!!\nмаксимальная скорость ракеты больше скорости звука в воздухе (335 м/с)\nпоэтому сопротивление воздуха может быть расчитано не точно\n(т.к. волновое сопротивление не учитывается)\n!!!\033[0m')
        print('максимальная скорость:'                                        , max_speed                                                 , 'м/с')
        print('максимальная перегрузка:'                                      , max_g                                                     , 'g')

        show(X      , Yp                         , 'Относительное давление'    , 'время, с', 'давление, атм')
        show(X[1:-1], [t2 - 273.14 for t2 in Yt2], 'Температура у выхода сопла', 'время, с', 'температура, °C')
        show(X      , Ykn            , 'Kn' + ['', ' (без трассера)'][l_t > 0.], 'время, с', 'Kn')
        if l_t > 0.: show(X, Ykn_t               , 'Kn (с трассером)'          , 'время, с', 'Kn')
        show(X      , Yt                         , 'Тяга'                      , 'время, с', 'тяга, кгс')
        show(X      , Ym                         , 'Cгорание топлива'          , 'время, с', 'масса оставшегося топлива, г')
        hn = len(Yt)#[Yh[i + 3] < Yh[i] for i in range(len(Yh) - 3)].index(True)
        show(Xh[:hn], Yh[:hn]                    , 'Полёт'                     , 'время, с', 'высота, м', Xh[hn:], Yh[hn:])
    
    return main
