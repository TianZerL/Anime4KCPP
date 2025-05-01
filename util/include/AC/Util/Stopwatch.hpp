#ifndef AC_UTIL_STOPWATCH_HPP
#define AC_UTIL_STOPWATCH_HPP

#include <cstdint>
#include <cstdio>
#include <chrono>

namespace ac::util
{
	class Stopwatch;
}

class ac::util::Stopwatch
{
public:
	using FormatBuffer = char[7];
public:
	Stopwatch() noexcept;
	void reset() noexcept;
	void stop() noexcept;
	double elapsed() noexcept;

	static char* formatDuration(FormatBuffer& buffer, double duration) noexcept;
private:
	bool stopFlag;
	std::chrono::time_point<std::chrono::steady_clock> start, end;
};

inline ac::util::Stopwatch::Stopwatch() noexcept
{
	reset();
}

inline void ac::util::Stopwatch::reset() noexcept
{
	stopFlag = false;
	end = start = std::chrono::steady_clock::now();
}
inline void ac::util::Stopwatch::stop() noexcept
{
	end = std::chrono::steady_clock::now();
	stopFlag = true;
}
inline double ac::util::Stopwatch::elapsed() noexcept
{
	return std::chrono::duration<double>{ (stopFlag ? end : std::chrono::steady_clock::now()) - start }.count();
}

inline char* ac::util::Stopwatch::formatDuration(FormatBuffer& buffer, const double duration) noexcept
{
	if (duration < 0) std::snprintf(buffer, sizeof(buffer), "#ERROR");
	else if (duration < 3600)
	{
		auto ss = static_cast<int>(duration + 0.5);
		int mm = ss / 60;
		ss %= 60;
		std::snprintf(buffer, sizeof(buffer), "%02dm%02ds", mm, ss);
	}
	else if (duration < 3600 * 24)
	{
		auto ss = static_cast<int>(duration + 0.5);
		int mm = ss / 60;
		int hh = mm / 60;
		mm %= 60;
		std::snprintf(buffer, sizeof(buffer), "%02dh%02dm", hh, mm);
	}
	else
	{
		auto ss = static_cast<std::int64_t>(duration + 0.5);
		std::int64_t hh = ss / 3600;
		std::int64_t dd = hh / 24;
		hh %= 24;
		if (dd < 99) std::snprintf(buffer, sizeof(buffer), "%02dd%02dh", static_cast<int>(dd), static_cast<int>(hh));
		else std::snprintf(buffer, sizeof(buffer), "99+day");
	}

	return buffer;
}

#endif
