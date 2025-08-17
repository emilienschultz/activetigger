import { FC } from 'react';
import { SubmitHandler, useForm } from 'react-hook-form';
import { useComputeBertTopic } from '../../../src/core/api';
import { ComputeBertTopicModel } from '../../../src/types';
interface BertTopicCreationFormProps {
  projectSlug: string | null;
}

export const BertTopicForm: FC<BertTopicCreationFormProps> = ({ projectSlug }) => {
  const { computeBertTopic } = useComputeBertTopic(projectSlug);
  const { handleSubmit: handleSubmitNewModel, register } = useForm<ComputeBertTopicModel>({
    defaultValues: {
      name: 'bertopic',
      outlier_reduction: true,
      min_topic_size: 10,
    },
  });

  const onSubmitNewModel: SubmitHandler<ComputeBertTopicModel> = async (data) => {
    await computeBertTopic(data);
  };
  return (
    <form onSubmit={handleSubmitNewModel(onSubmitNewModel)}>
      <label className="form-label" htmlFor="name">
        Name
        <input className="form-control" id="name" type="text" {...register('name')} />
      </label>
      <label className="form-label" htmlFor="min_topic_size">
        Min topic size
        <input
          className="form-control"
          id="minTopicSize"
          type="number"
          {...register('min_topic_size')}
        />
      </label>
      <label className="form-label" htmlFor="outlier_reduction">
        Outlier reduction
        <input id="outlierReduction" type="checkbox" {...register('outlier_reduction')} />
      </label>

      <button type="submit">Compute BertTopic on Trainset</button>
    </form>
  );
};
