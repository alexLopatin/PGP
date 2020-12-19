// nvcc opengl_2.cu -lGL -lGLU -lglut -lGLEW
// ./a.out

// Для linux нужно поставить пакеты: libgl1-mesa-dev libglew-dev freeglut3-dev
// sudo apt-get install libgl1-mesa-dev libglew-dev freeglut3-dev

//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"

#define _USE_MATH_DEFINES

#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <math.h>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

typedef unsigned char uchar;

#define sqr3(x) ((x)*(x)*(x))
#define sqr(x) ((x)*(x))

struct t_item {
	float x;
	float y;
	float z;
	float dx;
	float dy;
	float dz;
	float q;
};

const int N = 150;

t_item items[N];

t_item itemShot;

int w = 1024, h = 648;

float x = -1.5, y = -1.5, z = 1.0;
float dx = 0.0, dy = 0.0, dz = 0.0;
float yaw = 0.0, pitch = 0.0;
float dyaw = 0.0, dpitch = 0.0;

float qc = 300.0;			// заряд камеры
float qb = 50.0;			// заряд пули
float bspeed = 10.0;		// скорость пули

float speed = 0.2;
float g = 20.0;

#define K 50.0

#define a2 15.0
#define shift 0.75
const int np = 100;				// Размер текстуры пола

GLUquadric* quadratic;			// quadric объекты - это геометрические фигуры 2-го порядка, т.е. сфера, цилиндр, диск, конус. 

cudaGraphicsResource* res;
GLuint textures[2];				// Массив из текстурных номеров
GLuint vbo;						// Номер буфера


__global__ void kernel(uchar4* data, t_item* items, float t) {	// Генерация текстуры пола на GPU
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;
	int i, j;
	float x, y, f;
	for (i = idx; i < np; i += offsetx)
		for (j = idy; j < np; j += offsety) {
			f = 0.0;

			for (int k = 0; k < N; k++)
			{
				x = (2.0 * i / (np - 1.0) - 1.0) * a2;
				y = (2.0 * j / (np - 1.0) - 1.0) * a2;
				f += K * items[k].q / (sqr(x - items[k].x) + sqr(y - items[k].y) + sqr(items[k].z - shift) + 0.001);
				f = min(max(90.0f, f), 255.0f);
			}
			data[j * np + i] = make_uchar4(0, (int)f, 0, 255);
		}
}

bool shot = false;

void display() {
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	// Задаем "объектив камеры"
	gluPerspective(90.0f, (GLfloat)w / (GLfloat)h, 0.1f, 100.0f);


	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	// Задаем позицию и направление камеры
	gluLookAt(x, y, z,
		x + cos(yaw) * cos(pitch),
		y + sin(yaw) * cos(pitch),
		z + sin(pitch),
		0.0f, 0.0f, 1.0f);

	glBindTexture(GL_TEXTURE_2D, textures[0]);	// Задаем текстуру


	static float angle = 0.0;

	for (int i = 0; i < N; i++)
	{
		glPushMatrix();
		glTranslatef(items[i].x, items[i].y, items[i].z);	// Задаем координаты центра сферы
		glRotatef(angle, 0.0, 0.0, 1.0);
		gluSphere(quadratic, 0.5f, 32, 32);
		glPopMatrix();
	}
	
	if (shot)
	{
		glPushMatrix();
		glTranslatef(itemShot.x, itemShot.y, itemShot.z);
		gluSphere(quadratic, 0.5f, 32, 32);
		glPopMatrix();
	}

	angle += 0.15;


	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, vbo);	// Делаем активным буфер с номером vbo
	glBindTexture(GL_TEXTURE_2D, textures[1]);	// Делаем активной вторую текстуру
	glTexImage2D(GL_TEXTURE_2D, 0, 3, (GLsizei)np, (GLsizei)np, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);	// Деактивируем буфер
	// Последний параметр NULL в glTexImage2D говорит о том что данные для текстуры нужно брать из активного буфера

	glBegin(GL_QUADS);			// Рисуем пол
	glTexCoord2f(0.0, 0.0);
	glVertex3f(-a2, -a2, 0.0);

	glTexCoord2f(1.0, 0.0);
	glVertex3f(a2, -a2, 0.0);

	glTexCoord2f(1.0, 1.0);
	glVertex3f(a2, a2, 0.0);

	glTexCoord2f(0.0, 1.0);
	glVertex3f(-a2, a2, 0.0);
	glEnd();


	glBindTexture(GL_TEXTURE_2D, 0);			// Деактивируем текстуру

	// Отрисовка каркаса куба				
	glLineWidth(2);								// Толщина линий				
	glColor3f(0.5f, 0.5f, 0.5f);				// Цвет линий
	glBegin(GL_LINES);							// Последующие пары вершин будут задавать линии
	glVertex3f(-a2, -a2, 0.0);
	glVertex3f(-a2, -a2, 2.0 * a2);

	glVertex3f(a2, -a2, 0.0);
	glVertex3f(a2, -a2, 2.0 * a2);

	glVertex3f(a2, a2, 0.0);
	glVertex3f(a2, a2, 2.0 * a2);

	glVertex3f(-a2, a2, 0.0);
	glVertex3f(-a2, a2, 2.0 * a2);
	glEnd();

	glBegin(GL_LINE_LOOP);						// Все последующие точки будут соеденены замкнутой линией
	glVertex3f(-a2, -a2, 0.0);
	glVertex3f(a2, -a2, 0.0);
	glVertex3f(a2, a2, 0.0);
	glVertex3f(-a2, a2, 0.0);
	glEnd();

	glBegin(GL_LINE_LOOP);
	glVertex3f(-a2, -a2, 2.0 * a2);
	glVertex3f(a2, -a2, 2.0 * a2);
	glVertex3f(a2, a2, 2.0 * a2);
	glVertex3f(-a2, a2, 2.0 * a2);
	glEnd();

	glColor3f(1.0f, 1.0f, 1.0f);

	glutSwapBuffers();
}

t_item* deviceItems;

void update() {
	float v = sqrt(dx * dx + dy * dy + dz * dz);
	if (v > speed) {		// Ограничение максимальной скорости
		dx *= speed / v;
		dy *= speed / v;
		dz *= speed / v;
	}
	x += dx; dx *= 0.99;
	y += dy; dy *= 0.99;
	z += dz; dz *= 0.99;
	if (z < 1.0) {			// Пол, ниже которого камера не может переместиться
		z = 1.0;
		dz = 0.0;
	}
	if (fabs(dpitch) + fabs(dyaw) > 0.0001) {	// Вращение камеры
		yaw += dyaw;
		pitch += dpitch;
		pitch = fmin(M_PI / 2.0 - 0.0001, fmax(-M_PI / 2.0 + 0.0001, pitch));
		dyaw = dpitch = 0.0;
	}

	float w = 0.99, e0 = 1e-3, dt = 0.01;

	itemShot.x += itemShot.dx * dt;
	itemShot.y += itemShot.dy * dt;
	itemShot.z += itemShot.dz * dt;

	for (int i = 0; i < N; i++)
	{
		// Замедление
		items[i].dx *= w;
		items[i].dy *= w;
		items[i].dz *= w;

		for (int j = 0; j < N; j++)
		{
			float l = sqrt(sqr(items[i].x - items[j].x) + sqr(items[i].y - items[j].y) + sqr(items[i].z - items[j].z));
			items[i].dx += K * items[i].q * items[j].q * (items[i].x - items[j].x) / (l * l * l + e0) * dt;
			items[i].dy += K * items[i].q * items[j].q * (items[i].y - items[j].y) / (l * l * l + e0) * dt;
			items[i].dz += K * items[i].q * items[j].q * (items[i].z - items[j].z) / (l * l * l + e0) * dt;
		}

		// Отталкивание от стен
		items[i].dx += items[i].q * items[i].q * K * (items[i].x - a2) / (sqr3(fabs(items[i].x - a2)) + e0) * dt;
		items[i].dx += items[i].q * items[i].q * K * (items[i].x + a2) / (sqr3(fabs(items[i].x + a2)) + e0) * dt;

		items[i].dy += items[i].q * items[i].q * K * (items[i].y - a2) / (sqr3(fabs(items[i].y - a2)) + e0) * dt;
		items[i].dy += items[i].q * items[i].q * K * (items[i].y + a2) / (sqr3(fabs(items[i].y + a2)) + e0) * dt;

		items[i].dz += items[i].q * items[i].q * K * (items[i].z - 2 * a2) / (sqr3(fabs(items[i].z - 2 * a2)) + e0) * dt;
		items[i].dz += items[i].q * items[i].q * K * (items[i].z + 0.0) / (sqr3(fabs(items[i].z + 0.0)) + e0) * dt;

		items[i].dz -= g * dt;

		// Отталкивание от камеры
		float l = sqrt(sqr(items[i].x - x) + sqr(items[i].y - y) + sqr(items[i].z - z));
		items[i].dx += qc * items[i].q * K * (items[i].x - x) / (l * l * l + e0) * dt;
		items[i].dy += qc * items[i].q * K * (items[i].y - y) / (l * l * l + e0) * dt;
		items[i].dz += qc * items[i].q * K * (items[i].z - z) / (l * l * l + e0) * dt;

		// Отталкивание от пули
		l = sqrt(sqr(items[i].x - itemShot.x) + sqr(items[i].y - itemShot.y) + sqr(items[i].z - itemShot.z));
		items[i].dx += itemShot.q * items[i].q * K * (items[i].x - itemShot.x) / (l * l * l + e0) * dt;
		items[i].dy += itemShot.q * items[i].q * K * (items[i].y - itemShot.y) / (l * l * l + e0) * dt;
		items[i].dz += itemShot.q * items[i].q * K * (items[i].z - itemShot.z) / (l * l * l + e0) * dt;

		items[i].x += items[i].dx * dt;
		items[i].y += items[i].dy * dt;
		items[i].z += items[i].dz * dt;

		// чтобы шарики точно не вылетели
		items[i].x = fmin(fmax(items[i].x, -a2 + e0), a2 - e0);
		items[i].y = fmin(fmax(items[i].y, -a2 + e0), a2 - e0);
		items[i].z = fmin(fmax(items[i].z, e0), 2 * a2 - e0);
	}

	static float t = 0.0;
	uchar4* dev_data;
	size_t size;
	cudaMemcpy(deviceItems, items, sizeof(t_item) * N, cudaMemcpyHostToDevice);
	cudaGraphicsMapResources(1, &res, 0);		// Делаем буфер доступным для CUDA
	cudaGraphicsResourceGetMappedPointer((void**)&dev_data, &size, res);	// Получаем указатель на память буфера
	kernel << <dim3(32, 32), dim3(32, 8) >> > (dev_data, deviceItems, t);
	cudaGraphicsUnmapResources(1, &res, 0);		// Возращаем буфер OpenGL'ю что бы он мог его использовать
	t += 0.01;

	glutPostRedisplay();	// Перерисовка
}

void keys(unsigned char key, int x, int y) {	// Обработка кнопок
	switch (key) {
	case 'w':                 // "W" Движение вперед
		dx += cos(yaw) * cos(pitch) * speed;
		dy += sin(yaw) * cos(pitch) * speed;
		dz += sin(pitch) * speed;
		break;
	case 's':                 // "S" Назад
		dx += -cos(yaw) * cos(pitch) * speed;
		dy += -sin(yaw) * cos(pitch) * speed;
		dz += -sin(pitch) * speed;
		break;
	case 'a':                 // "A" Влево
		dx += -sin(yaw) * speed;
		dy += cos(yaw) * speed;
		break;
	case 'd':                 // "D" Вправо
		dx += sin(yaw) * speed;
		dy += -cos(yaw) * speed;
		break;
	case 27:
		cudaGraphicsUnregisterResource(res);
		glDeleteTextures(2, textures);
		glDeleteBuffers(1, &vbo);
		gluDeleteQuadric(quadratic);
		exit(0);
		break;
	}
}

void mouse(int x, int y) {
	static int x_prev = w / 2, y_prev = h / 2;
	float dx = 0.005 * (x - x_prev);
	float dy = 0.005 * (y - y_prev);
	dyaw -= dx;
	dpitch -= dy;
	x_prev = x;
	y_prev = y;

	// Перемещаем указатель мышки в центр, когда он достиг границы
	if ((x < 20) || (y < 20) || (x > w - 20) || (y > h - 20)) {
		glutWarpPointer(w / 2, h / 2);
		x_prev = w / 2;
		y_prev = h / 2;
	}
}

void mouseButton(int button, int state, int _x, int _y) {
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN)
	{
		shot = true;
		itemShot.x = x;
		itemShot.y = y;
		itemShot.z = z;

		itemShot.dx = cos(yaw) * cos(pitch) * bspeed;
		itemShot.dy = sin(yaw) * cos(pitch) * bspeed;
		itemShot.dz = sin(pitch) * bspeed;

		itemShot.q = qb;
	}
}

void reshape(int w_new, int h_new) {
	w = w_new;
	h = h_new;
	glViewport(0, 0, w, h);                                     // Сброс текущей области вывода
	glMatrixMode(GL_PROJECTION);                                // Выбор матрицы проекций
	glLoadIdentity();                                           // Сброс матрицы проекции
}

float frand()
{
	return static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
}

int main(int argc, char** argv) {
	srand(time(0));

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowSize(w, h);
	glutCreateWindow("OpenGL");

	glutIdleFunc(update);
	glutDisplayFunc(display);
	glutKeyboardFunc(keys);
	glutPassiveMotionFunc(mouse);
	glutMouseFunc(mouseButton);
	glutReshapeFunc(reshape);

	glutSetCursor(GLUT_CURSOR_NONE);	// Скрываем курсор мышки

	int wt, ht;
	FILE* in = fopen("in.data", "rb");
	fread(&wt, sizeof(int), 1, in);
	fread(&ht, sizeof(int), 1, in);
	uchar* data = (uchar*)malloc(sizeof(uchar) * wt * ht * 4);
	fread(data, sizeof(uchar), 4 * wt * ht, in);
	fclose(in);

	glGenTextures(2, textures);
	glBindTexture(GL_TEXTURE_2D, textures[0]);
	glTexImage2D(GL_TEXTURE_2D, 0, 3, (GLsizei)wt, (GLsizei)ht, 0, GL_RGBA, GL_UNSIGNED_BYTE, (void*)data);
	// если полигон, на который наносим текстуру, меньше текстуры
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); //GL_LINEAR);	// Интерполяция
	// если больше
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST); //GL_LINEAR);		


	quadratic = gluNewQuadric();
	gluQuadricTexture(quadratic, GL_TRUE);

	glBindTexture(GL_TEXTURE_2D, textures[1]);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);	// Интерполяция 
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);	// Интерполяция	

	glEnable(GL_TEXTURE_2D);                             // Разрешить наложение текстуры
	glShadeModel(GL_SMOOTH);                             // Разрешение сглаженного закрашивания
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);                // Черный фон
	glClearDepth(1.0f);                                  // Установка буфера глубины
	glDepthFunc(GL_LEQUAL);                              // Тип теста глубины. 
	glEnable(GL_DEPTH_TEST);                			 // Включаем тест глубины
	glEnable(GL_CULL_FACE);                 			 // Режим при котором, тектуры накладываются только с одной стороны

	glewInit();
	glGenBuffers(1, &vbo);								// Получаем номер буфера
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, vbo);			// Делаем его активным
	glBufferData(GL_PIXEL_UNPACK_BUFFER, np * np * sizeof(uchar4), NULL, GL_DYNAMIC_DRAW);	// Задаем размер буфера
	cudaGraphicsGLRegisterBuffer(&res, vbo, cudaGraphicsMapFlagsWriteDiscard);				// Регистрируем буфер для использования его памяти в CUDA
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);			// Деактивируем буфер

	for (int i = 0; i < N; i++)
	{
		items[i].x = frand() * 2 * a2 - a2;
		items[i].y = frand() * 2 * a2 - a2;
		items[i].z = frand() * 2 * a2;
		items[i].dx = items[i].dy = items[i].dz = 0;
		items[i].q = 1.0;
	}

	cudaMalloc(&deviceItems, sizeof(t_item) * N);

	glutMainLoop();
}
