import datetime


def get_date_time(str_date, str_time):
    """
    Global Function: getDataTime(str_date, str_time):
    Description:
        Take Data String and Time String, spinning out a
        datetime object corresponding to the date and time
        provided.
    :param str_date: Date String D-M-Y
    :param str_time: Time String with Format H:M:S
    :return datetime: Converting Date, Time string into a
        datetime variable
    """
    data_list = str_date.split('-')
    time_list = str_time.split(':')
    sec_list = time_list[2].split('.')
    dt = datetime.datetime(int(data_list[0]),
                           int(data_list[1]),
                           int(data_list[2]),
                           int(time_list[0]),
                           int(time_list[1]),
                           int(sec_list[0]),
                           int(sec_list[1]))
    return dt
