import { FC } from 'react';
import { SubmitHandler, useForm } from 'react-hook-form';
import { useComputeBertopic } from '../../core/api';
import { ComputeBertopicModel } from '../../types';
interface BertopicCreationFormProps {
  projectSlug: string | null;
}

export const BertopicForm: FC<BertopicCreationFormProps> = ({ projectSlug }) => {
  const { computeBertopic } = useComputeBertopic(projectSlug);
  const { handleSubmit: handleSubmitNewModel, register } = useForm<ComputeBertopicModel>({
    defaultValues: {
      name: 'bertopic',
      outlier_reduction: true,
      min_topic_size: 10,
      nr_topics: 'auto',
    },
  });

  const onSubmitNewModel: SubmitHandler<ComputeBertopicModel> = async (data) => {
    await computeBertopic(data);
  };
  return (
    <form onSubmit={handleSubmitNewModel(onSubmitNewModel)}>
      <label className="form-label" htmlFor="name">
        Name
        <input className="form-control" id="name" type="text" {...register('name')} />
      </label>
      <label className="form-label" htmlFor="outlier_reduction">
        Outlier reduction
        <input
          id="outlierReduction"
          type="checkbox"
          {...register('outlier_reduction')}
          className="mx-2"
        />
      </label>
      <details className="custom-details">
        <summary>Advanced parameters</summary>
        <label className="form-label" htmlFor="min_topic_size">
          Min topic size
          <input
            className="form-control"
            id="minTopicSize"
            type="number"
            {...register('min_topic_size')}
          />
        </label>
        <label className="form-label" htmlFor="nr_topics">
          Number of topics (optional)
          <input className="form-control" id="nr_topics" type="number" {...register('nr_topics')} />
        </label>
      </details>

      <button className="btn btn-primary w-50">Compute Bertopic on trainset</button>
    </form>
  );
};
