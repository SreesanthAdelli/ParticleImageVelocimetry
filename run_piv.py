from piv_processor import PIVProcessor

def main():
    piv = PIVProcessor("frame_1.tif", "frame_2.tif", win_size=32)
    piv.compute_velocity_field()
    piv.plot_velocity_field()

if __name__ == "__main__":
    main()
