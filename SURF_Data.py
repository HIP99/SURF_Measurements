import pickle
import numpy as np
import matplotlib.pyplot as plt
from pypdf import PdfWriter

class SURFData():
    """
    Pretty much all of this was written by Taylor and Payton
    Simply extracts the data from the SURFs pickle file
    Can save all the data to a pdf
    """
    def __init__(self, filepath, *args, **kwargs):

        self.filepath = filepath

        # self.format()

    @staticmethod
    def load_pkl_file(file_path):
        """Loads the pickle file"""
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        return data
    
    def format_data(self):
        """
        Returns all the SURF data for 28 SURFS with 8 Channels each. Each channel capture is 1024 samples
        This is not stored as a field because a lot of this data is not in use. Sub classes will handpick the data they want
        """
        ## Data is in 449 fragments
        loaded_data = self.load_pkl_file(self.filepath)

        data = np.empty((0))
        for i in range(len(loaded_data)):
            ## First 8 bytes of each fragments is the header, strip that off (offset = 8) during byte reading and concatenation
            data = np.concatenate((data, np.frombuffer(loaded_data[i], dtype = np.int16, offset = 8)))

        ## First 128 bytes of data is more headers, remove it
        ## There are 1024 samples per capture, 8 channels on a surf, 28 stuff taking data
        data_shaped = np.reshape(data[128:], (28, 8, 1024))


        ##Surf 20 and 27 are not SURF channels
        ##6 and 13 are LF SURFs
        return data_shaped
    
    def plot_all(self, ax: plt.Axes=None):
        """
        This plots all the data with blue lines separating SURFs and red lines separating Channels
        Useful if you don't know what channels to look at
        """
        if ax is None:
            fig, ax = plt.subplots()

        all_data = self.format_data()

        flat_data = all_data.flatten()

        samples = np.arange(0, 8*28*1024, 1)

        ax.plot(samples, flat_data)

        tick_arr = np.arange(8*512, 8*28*1024, 8*1024)

        ax.set_xticks(tick_arr, self.surf_mapping)


        for i in range(1,28):
            for j in range(8):
                ax.axvline(x = (i*8+j)*1024, alpha = 0.1 , color = 'blue', linestyle = '--')
            ax.axvline(x = i*8*1024, alpha = 0.4, color = 'red', linestyle = 'dashdot')

        ax.set_xlabel('Samples')
        ax.set_ylabel('Raw ADC counts')
        ax.set_title('SURFS total output')


    def save_pdf(self, file_name:str):
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

        merger.write(f"/Users/hpumphrey/Com/SURF_Measurements/results/{file_name}.pdf")
        merger.close()
    
if __name__ == '__main__':
    run = 0

    # offsets = np.arange(-200, 200, 20)
    offsets = np.arange(-50, 50, 10)
    print(offsets)

    from pathlib import Path
    current_dir = Path(__file__).resolve()

    parent_dir = current_dir.parents[1]

    fig, ax = plt.subplots()

    filename = 'hpolplease'

    # for offset in offsets:
    #     filepath = parent_dir / 'data' / filename / f'{filename}off{offset}_{run}.pkl'
    #     pckl = SURF_Data(filepath = filepath)
    #     pckl.save_pdf(f'hpolpleaseoff{offset}_{run}')


    filepath = parent_dir / 'data' / filename / f'{filename}off{offsets[3]}_{run}.pkl'
    pckl = SURFData(filepath = filepath)

    pckl.plot_all(ax=ax)

    # plt.legend()
    plt.show()