import pickle
import numpy as np
import matplotlib.pyplot as plt
from pypdf import PdfWriter

class SURF_Data():
    """
    Pretty much all of this was written by Taylor and Payton
    Simply extracts the data from the SURFs pickle file
    Can save all the data to a pdf
    """
    def __init__(self, filepath, *args, **kwargs):

        self.channel_mapping = [4,5,6,7,0,1,2,3]
        self.surf_mapping = ['FH', 'EH', 'DH', 'CH', 'BH', 'AH', 'LFV', 'GH', 'IH', 'JH', 'KH', 'LH', 'MH', 'LFH', 'GV', 'IV', 'JV', 'KV', 'LV', 'MV', None, 'FV', 'EV', 'DV', 'CV', 'BV', 'AV', None]

        self.filepath = filepath

        # self.format()

    def get_surf_index(self, surf_name):
        surf_name = surf[:-1]
        channel_num = int(surf[-1])

        surf_index = self.surf_mapping.index(surf_name)
        channel_index = self.channel_mapping[channel_num-1]

        return surf_index, channel_index

    @staticmethod
    def load_pkl_file(file_path):
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        return data
    
    def format_data(self):
        ## Data is in 449 fragments
        loaded_data = self.load_pkl_file(self.filepath)

        data = np.empty((0))
        for i in range(len(loaded_data)):
            ## First 8 bytes of each fragments is the header, strip that off (offset = 8) during byte reading and concatenation
            data = np.concatenate((data, np.frombuffer(loaded_data[i], dtype = np.int16, offset = 8)))

        ## First 128 bytes of data is more headers, remove it
        ## There are 1024 samples per capture, 8 channels on a surf, 28 stuff taking data
        data_shaped = np.reshape(data[128:], (28, 8, 1024))

        ##Surf 6, 13, 20 and 27 are not SURF channels
        return data_shaped

    def save_pdf(self):
        self.data = self.format_data()
        ## Plot all the different surfs and channels on seperate graphs
        fn = []
        for ii in range(28):
            fig, axs = plt.subplots(nrows = 4, ncols = 2, figsize=(10,10))

            plt.suptitle('SURF ' + str(ii), fontsize = 15, fontweight = 'bold')
            for i in range(8):
                if (i%2==0):
                    axs[i//2, 0].plot(self.data[ii, i])
                    axs[i//2, 0].set_title('Channel ' + str(i))
                    axs[i//2, 0].set_xlabel('Sample Number')
                    axs[i//2, 0].set_ylabel('ADC Count')
                    axs[i//2, 0].set_ylim(ymin=-1000, ymax=1000)

                else:
                    axs[(i-1)//2, 1].plot(self.data[ii, i])
                    axs[(i-1)//2, 1].set_title('Channel ' + str(i))
                    axs[(i-1)//2, 1].set_xlabel('Sample Number')
                    axs[(i-1)//2, 1].set_ylabel('ADC Count')
                    axs[(i-1)//2, 1].set_ylim(ymin=-1000, ymax=1000)
                
            plt.tight_layout()
            plt.ylim((-900, 900))
            filename = 'SURF_' + str(ii) + 'data.pdf'
            fig.savefig(filename)
            fn.append(filename)

        merger = PdfWriter()

        for pdf in fn:
            merger.append(pdf)

        merger.write("AMPAt34.pdf")
        merger.close()
    

if __name__ == '__main__':
    from pathlib import Path
    current_dir = Path(__file__).resolve()

    parent_dir = current_dir.parents[1]

    fig, ax = plt.subplots()

    surf = "GV3"

    file_path = parent_dir / 'data' / 'SURF_Data' / f'SURF{surf}' / f'SURF{surf}_1.pkl'
    pckl = SURF_Data(filepath = file_path)

    # plt.legend()
    # plt.show()